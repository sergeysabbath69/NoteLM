# WAL — Knowledge Studio (NoteLM → v2)

## Статус: 🔄 В работе

## Этапы по ТЗ

### Этап 1: Инфраструктура и архитектура
- [ ] Переименовать NoteLM → Knowledge Studio
- [ ] Обновить модель данных: Projects → Sources → Chunks → Embeddings → Links
- [ ] Настроить async job queue (celery или asyncio)
- [ ] Подключить векторную БД (Qdrant или pgvector)
- [ ] JWT auth + row-level security

### Этап 2: Ingestion Pipeline
- [ ] PDF, DOCX, TXT, MD — уже есть частично
- [ ] Специальный тип КНИГА: TOC, главы, summary по главам
- [ ] Специальный тип БРЕНДБУК: извлечение brand_profile JSON
- [ ] URL ingestion с scraping
- [ ] YouTube транскрипция
- [ ] Семантическая индексация (chunking → embeddings → vector store)

### Этап 3: Семантические связи
- [ ] Автоматический поиск связей между чанками разных источников
- [ ] Хранение: source_a, source_b, link_type, confidence, explanation
- [ ] Типы связей: подтверждает / противоречит / развивает / иллюстрирует

### Этап 4: Генерации
- [ ] Чат с цитатами из источников
- [ ] Презентации (PPTX)
- [ ] Инфографика (Imagen 4)
- [ ] Mind Map (граф)
- [ ] Study Guide, FAQ, Briefing, Timeline, Glossary
- [ ] **ПОДКАСТЫ** — ключевая фича
  - [ ] Общий по книге
  - [ ] По каждой главе
  - [ ] Тематический по набору документов

### Этап 5: UI
- [ ] 3-панельный layout (текущий — хорошая база)
- [ ] Knowledge Map (граф связей между источниками) — React Flow / D3
- [ ] Дерево структуры книги + навигация по главам
- [ ] Прогресс-индикаторы на все тяжёлые операции
- [ ] Тёмная тема

## Решения принятые

- Векторная БД: pgvector (уже есть PostgreSQL контекст) или Qdrant
- Очереди: asyncio + SQLite job store (для MVP), Celery для prod
- TTS: gTTS (текущий) + Gemini TTS (для quality)
- Backend: FastAPI (текущий) — расширяем
- LLM: Gemini 2.5 Flash + Mistral fallback

## Открытые вопросы для Sergey

1. Брендбук по умолчанию — Центр-инвест или нейтральный?
2. Хранение данных — только локально или S3?
3. Приоритет: подкасты vs граф знаний vs чат?
4. Нужна ли аутентификация в MVP (один пользователь или multi-user)?
