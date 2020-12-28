import os
import config
import aiogram
import neuro
from neuro import VAE
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types.message import ContentType
from aiogram.utils.markdown import text, italic
from aiogram.types import ParseMode
import uuid

print('start')

bot = Bot(token=config.token)
dispatcher = Dispatcher(bot)

@dispatcher.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет!\nЭто сервис для преобразования фото в аватар."
                        "Пожалуйста загрузите селфи.\nПодбробнее: /help")

@dispatcher.message_handler(commands=['help'])
async def process_start_command(message: types.Message):
    await message.reply(text("Этот бот обучен на датасете лиц, с помощью pytorch на архитектуре автоэнкодеров.\n",
                        "Входящая картинка будет сжата до 45х45, и пропущена через автоэнкодер.\n",
                        "На выходе будет восстановлена картинка 45х45 с обобщенным изображением лица.\n",
                        "__Любая входящая картинка будет преобразована в лицо__.\n",
                        "Для улучшение качества восстановления **рекомендуется** загружать\n",
                        "квадратное фото лица расположенное анфас без наклонов,\n",
                        "с носом примерно посередине изображения.\n",
                        "Vladimir Zenin, 2020, @vladimirzenin\n",
                        "https://github.com/vladimirzenin/made_ml_hw4"))

@dispatcher.message_handler(content_types=ContentType.PHOTO)
async def photo_message(message: types.Message):
    # Обрабатываем только 1 фото, даже если пришел массив.
    # -1 потому что в конце массива гарантированно самый большой файл.
    filename = 'inp_' + str(uuid.uuid4()) + '.jpg'
    await message.photo[-1].download(filename)
    answ_filename = neuro.getRecon(filename)
    with open(answ_filename, 'rb') as file:
        await bot.send_photo(message.from_user.id, file,
                             caption="Обработано",
                             reply_to_message_id=message.message_id)

@dispatcher.message_handler(content_types=ContentType.ANY)
async def unknown_message(message: types.Message):
    message_text = text("Команда не поддерживается\n",
                        italic("Подробнее о боте: "), "/help")
    await message.reply(message_text, parse_mode=ParseMode.MARKDOWN)

if __name__ == "__main__":
    print('start_polling')
    neuro.init()
    print('start_polling')
    executor.start_polling(dispatcher)
