"""
Telegram-бот «универсальный помощник» с двумя типами памяти.

Короткая память  — последние 10 сообщений пользователя и 10 ответов бота (в оперативной памяти).
Долгая память    — загруженный документ (PDF / TXT / DOCX) → чанки → эмбеддинги → ChromaDB (./memory).

Документ задаёт роль помощника и служит базой знаний. Новый документ полностью
заменяет старый: помощник получает новую роль и новую базу знаний.

Установка зависимостей:
    pip install -r requirements.txt

Переменные окружения (можно задать в файле .env рядом со скриптом):
    BOT_TOKEN       — токен Telegram-бота
    PPROXY_API_KEY  — ключ ProxyAPI

Запуск:
    python bot_two_types_of_memory.py
"""

import asyncio
import logging
import os
import re
from collections import defaultdict, deque
from pathlib import Path

import chromadb
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

# Подгружаем переменные из .env рядом со скриптом (уже заданные в системе не перезаписываются)
load_dotenv(Path(__file__).with_name(".env"))

BOT_TOKEN = os.getenv("BOT_TOKEN")
PROXY_API_KEY = os.getenv("PPROXY_API_KEY")

PROXY_API_BASE_URL = "https://api.proxyapi.ru/v1"
CHAT_MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

MEMORY_DIR = "./memory"             # persistent-хранилище ChromaDB
DOCUMENTS_DIR = "./documents"       # сюда сохраняются загруженные файлы
HISTORY_SIZE = 10                   # размер короткой памяти: сколько пар «вопрос + ответ» помнить
CHUNK_SIZE = 500                    # размер чанка в символах
CHUNK_OVERLAP = 100                 # перекрытие чанков, чтобы не терять смысл на границах
TOP_K = 4                           # сколько фрагментов доставать из векторной базы
EMBED_BATCH_SIZE = 100              # сколько чанков отправлять в API эмбеддингов за раз
ROLE_SOURCE_CHARS = 4000            # сколько символов начала документа использовать для определения роли
MAX_FILE_SIZE = 20 * 1024 * 1024    # лимит Telegram Bot API на скачивание файлов
TELEGRAM_MSG_LIMIT = 4096

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

DEFAULT_ROLE = (
    "Ты — универсальный дружелюбный помощник. Отвечай кратко, по делу и на языке пользователя."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("two_memory_bot")

if not BOT_TOKEN:
    raise SystemExit("Не задана переменная окружения BOT_TOKEN")
if not PROXY_API_KEY:
    raise SystemExit("Не задана переменная окружения PPROXY_API_KEY")

Path(MEMORY_DIR).mkdir(parents=True, exist_ok=True)
Path(DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# КЛИЕНТЫ
# ============================================================================

# ProxyAPI полностью совместим с OpenAI SDK — меняется только base_url
llm = AsyncOpenAI(api_key=PROXY_API_KEY, base_url=PROXY_API_BASE_URL)

# Локальная векторная база с хранением на диске
chroma = chromadb.PersistentClient(path=MEMORY_DIR)

# Короткая память: user_id -> deque последних N сообщений {"role": ..., "content": ...}
# Каждый обмен репликами занимает 2 записи (user + assistant), поэтому maxlen = HISTORY_SIZE * 2
short_memory: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_SIZE * 2))


def collection_name(user_id: int) -> str:
    """У каждого пользователя своя коллекция = своя долгая память и своя роль."""
    return f"user_{user_id}"


def get_user_collection(user_id: int):
    """Возвращает коллекцию пользователя или None, если документ ещё не загружен."""
    try:
        return chroma.get_collection(collection_name(user_id))
    except Exception:
        return None


def get_user_role(user_id: int) -> str:
    """Роль хранится в метаданных коллекции, поэтому переживает перезапуск бота."""
    collection = get_user_collection(user_id)
    if collection is not None and collection.metadata:
        return collection.metadata.get("role", DEFAULT_ROLE)
    return DEFAULT_ROLE


# ============================================================================
# ДОЛГАЯ ПАМЯТЬ: загрузка, разбиение, эмбеддинги, поиск
# ============================================================================

def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    # Таблицы в DOCX тоже часто содержат важную информацию
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _read_txt(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Разбивает текст на чанки ≈ chunk_size символов с перекрытием.
    Старается резать по границе абзаца / предложения / пробела, а не посреди слова.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            # Ищем «красивую» точку разреза во второй половине окна
            for separator in ("\n\n", "\n", ". ", "! ", "? ", "; ", " "):
                cut = window.rfind(separator)
                if cut > chunk_size // 2:
                    end = start + cut + len(separator)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_document(path: str | Path) -> tuple[str, list[str]]:
    """
    Шаг 1 долгой памяти: читает файл (PDF / TXT / DOCX) и разбивает его на чанки.
    Возвращает (полный текст, список чанков).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _read_pdf(path)
    elif ext == ".docx":
        text = _read_docx(path)
    elif ext == ".txt":
        text = _read_txt(path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    if not text.strip():
        raise ValueError("Не удалось извлечь текст из документа (возможно, это скан без текстового слоя).")

    return text, split_into_chunks(text)


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Получает эмбеддинги через ProxyAPI пачками."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = await llm.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors


async def extract_role(text: str) -> str:
    """
    Просит модель сформулировать системную роль помощника на основе начала документа.
    Документ может явно описывать роль («Ты — консультант магазина…»), а может быть
    просто базой знаний — тогда роль выводится из тематики.
    """
    prompt = (
        "Ниже начало документа, который загрузили в AI-помощника. Документ может содержать "
        "явное описание роли помощника и/или справочную информацию.\n"
        "Сформулируй системную инструкцию для помощника во втором лице («Ты — …»): кто он, "
        "для кого работает, каким тоном и стилем общается, какие у него правила. Если роль "
        "явно описана в документе — сохрани её максимально точно. Если нет — выведи "
        "подходящую роль эксперта по теме документа. Не более 1200 символов, только текст инструкции.\n\n"
        f"=== ДОКУМЕНТ ===\n{text[:ROLE_SOURCE_CHARS]}"
    )
    try:
        response = await llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        role = (response.choices[0].message.content or "").strip()
        return role or DEFAULT_ROLE
    except Exception:
        logger.exception("Не удалось определить роль по документу")
        return DEFAULT_ROLE


async def embed_chunks(user_id: int, chunks: list[str], role: str, source_name: str) -> int:
    """
    Шаг 2 долгой памяти: превращает чанки в эмбеддинги и сохраняет в ChromaDB.
    Старая коллекция пользователя удаляется — новый документ = новая роль и новые знания.
    Возвращает количество сохранённых чанков.
    """
    vectors = await get_embeddings(chunks)

    def _store() -> None:
        name = collection_name(user_id)
        try:
            chroma.delete_collection(name)
        except Exception:
            pass  # коллекции ещё не было
        collection = chroma.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "role": role, "source": source_name},
        )
        collection.add(
            ids=[f"{user_id}_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors,
            metadatas=[{"source": source_name, "chunk": i} for i in range(len(chunks))],
        )

    # Chroma синхронная — выносим в поток, чтобы не блокировать event loop
    await asyncio.to_thread(_store)
    return len(chunks)


async def retrieve_context(user_id: int, query: str, top_k: int = TOP_K) -> list[str]:
    """
    Шаг 3 долгой памяти: ищет в ChromaDB фрагменты, наиболее близкие к вопросу.
    """
    collection = await asyncio.to_thread(get_user_collection, user_id)
    if collection is None:
        return []
    count = await asyncio.to_thread(collection.count)
    if count == 0:
        return []

    query_vector = (await get_embeddings([query]))[0]
    result = await asyncio.to_thread(
        collection.query,
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
    )
    return result.get("documents", [[]])[0]


# ============================================================================
# ГЕНЕРАЦИЯ ОТВЕТА (короткая + долгая память)
# ============================================================================

async def answer_question(user_id: int, question: str) -> str:
    """
    Собирает запрос к модели:
      system  — роль из документа + найденный контекст + правила «не выдумывать»;
      history — короткая память (последние 10 пар «вопрос + ответ»);
      user    — текущий вопрос.
    Затем сохраняет вопрос и ответ в короткую память.
    """
    role = await asyncio.to_thread(get_user_role, user_id)
    context_chunks = await retrieve_context(user_id, question)

    if context_chunks:
        context = "\n\n---\n\n".join(context_chunks)
        system_prompt = (
            f"{role}\n\n"
            "У тебя есть долгая память — фрагменты загруженного документа (ниже). "
            "Правила:\n"
            "1. На вопросы по теме документа отвечай СТРОГО на основе этих фрагментов.\n"
            "2. Если нужной информации во фрагментах нет — честно скажи, что в документе "
            "этого нет. Не выдумывай факты, цифры, имена.\n"
            "3. Учитывай историю диалога, чтобы понимать уточняющие вопросы.\n"
            "4. На приветствия и бытовые реплики отвечай естественно, оставаясь в своей роли.\n\n"
            f"=== ФРАГМЕНТЫ ДОКУМЕНТА ===\n{context}\n=== КОНЕЦ ФРАГМЕНТОВ ==="
        )
    else:
        system_prompt = role

    history = list(short_memory[user_id])
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": question}]

    response = await llm.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    answer = (response.choices[0].message.content or "").strip() or "Не удалось сформировать ответ."

    # Обновляем короткую память: deque(maxlen=10) сам выкидывает старые сообщения
    short_memory[user_id].append({"role": "user", "content": question})
    short_memory[user_id].append({"role": "assistant", "content": answer})
    return answer


# ============================================================================
# TELEGRAM-ОБРАБОТЧИКИ
# ============================================================================

dp = Dispatcher()

HELP_TEXT = (
    "Я универсальный помощник с двумя типами памяти.\n\n"
    "• Короткая память — помню ваши последние 10 сообщений и мои ответы на них.\n"
    "• Долгая память — пришлите файл PDF, TXT или DOCX с описанием моей роли и нужной "
    "информацией. Я изучу его, приму роль и буду отвечать на вопросы по документу.\n\n"
    "Новый документ заменяет предыдущий — вместе с ролью.\n\n"
    "Команды:\n"
    "/upload — как загрузить документ\n"
    "/role — показать текущую роль\n"
    "/reset — очистить короткую память (историю диалога)\n"
    "/forget — удалить документ и вернуться к роли по умолчанию\n"
    "/help — эта справка"
)


async def send_long(message: Message, text: str) -> None:
    """Telegram ограничивает сообщение 4096 символами — режем длинные ответы."""
    for i in range(0, len(text), TELEGRAM_MSG_LIMIT):
        await message.answer(text[i:i + TELEGRAM_MSG_LIMIT])


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer("Привет! " + HELP_TEXT)


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@dp.message(Command("upload"))
async def cmd_upload(message: Message) -> None:
    await message.answer(
        "Отправьте мне файл (PDF, TXT или DOCX) как документ. "
        "В начале документа желательно описать мою роль, дальше — всю информацию, "
        "которую нужно помнить."
    )


@dp.message(Command("role"))
async def cmd_role(message: Message) -> None:
    role = await asyncio.to_thread(get_user_role, message.from_user.id)
    await send_long(message, f"Моя текущая роль:\n\n{role}")


@dp.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    short_memory.pop(message.from_user.id, None)
    await message.answer("Короткая память очищена. Документ и роль сохранены.")


@dp.message(Command("forget"))
async def cmd_forget(message: Message) -> None:
    user_id = message.from_user.id
    try:
        await asyncio.to_thread(chroma.delete_collection, collection_name(user_id))
    except Exception:
        pass
    short_memory.pop(user_id, None)
    await message.answer("Долгая и короткая память очищены. Я снова универсальный помощник.")


@dp.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    """Загрузка документа → сохранение → чанки → роль → эмбеддинги → ChromaDB."""
    user_id = message.from_user.id
    document = message.document
    file_name = document.file_name or "document"
    ext = Path(file_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        await message.answer("Поддерживаются только файлы PDF, TXT и DOCX.")
        return
    if document.file_size and document.file_size > MAX_FILE_SIZE:
        await message.answer("Файл слишком большой: Telegram позволяет ботам скачивать файлы до 20 МБ.")
        return

    status = await message.answer("📥 Получил документ, обрабатываю…")
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        # 1. Сохраняем файл на диск
        user_dir = Path(DOCUMENTS_DIR) / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w.\-]", "_", file_name)
        local_path = user_dir / safe_name
        await bot.download(document, destination=local_path)

        # 2. Извлекаем текст и режем на чанки (парсинг PDF может быть долгим — в отдельном потоке)
        text, chunks = await asyncio.to_thread(load_document, local_path)
        await status.edit_text(f"✂️ Текст разбит на {len(chunks)} фрагментов. Определяю роль…")

        # 3. Определяем новую роль по документу
        role = await extract_role(text)
        await status.edit_text("🧠 Создаю эмбеддинги и сохраняю в долгую память…")

        # 4. Эмбеддинги → ChromaDB (старый документ и роль заменяются)
        saved = await embed_chunks(user_id, chunks, role, file_name)

        # Новая роль — новый диалог: старая история может противоречить роли
        short_memory.pop(user_id, None)

        await status.edit_text(
            f"✅ Документ «{file_name}» загружен в долгую память ({saved} фрагментов).\n"
            "Короткая память очищена. Можете задавать вопросы!"
        )
        await send_long(message, f"Моя новая роль:\n\n{role}")
    except ValueError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception:
        logger.exception("Ошибка обработки документа")
        await status.edit_text("❌ Не удалось обработать документ. Попробуйте другой файл или позже.")


@dp.message(F.text)
async def handle_text(message: Message, bot: Bot) -> None:
    """Обычное сообщение: короткая память + поиск в долгой памяти → ответ модели."""
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        answer = await answer_question(message.from_user.id, message.text)
    except Exception:
        logger.exception("Ошибка генерации ответа")
        answer = "❌ Произошла ошибка при обращении к модели. Попробуйте ещё раз чуть позже."
    await send_long(message, answer)


@dp.message()
async def handle_other(message: Message) -> None:
    await message.answer("Я понимаю текстовые сообщения и документы PDF / TXT / DOCX.")


# ============================================================================
# ЗАПУСК
# ============================================================================

async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    logger.info("Бот запущен. Долгая память: %s", os.path.abspath(MEMORY_DIR))
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
