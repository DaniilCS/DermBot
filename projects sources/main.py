from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.utils.executor import start_polling
from aiogram.dispatcher import filters
import logging
import aioschedule as schedule
import asyncio
from model_predictions import predict_disease, predict_state, make_detection
from database_process import find_all_users_to_remind
from aiogram import Bot, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from database_process import registrate_user, check_if_registrated
from process_predictions import prepare_history, make_help

API_TOKEN = '5129129098:AAFzGZUuWRaEz8mKoFvZj4ai4xQ_IUL-5qM'

button_help = KeyboardButton('Помощь')
button_classify_disease = KeyboardButton('Классификация заболевания')
button_classify_state = KeyboardButton('Классификация состояние кожи')
button_detect_disease = KeyboardButton('Обнаружение заболеваний')
button_history = KeyboardButton('История')

greet_kb = ReplyKeyboardMarkup(resize_keyboard=True)
greet_kb.add(button_help)
greet_kb.add(button_classify_disease)
greet_kb.add(button_classify_state)
greet_kb.add(button_detect_disease)
greet_kb.add(button_history)

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

class PhotoForClassifyDisease(StatesGroup):
    disease = State()
    state = State()
    detect = State()

async def on_startup(_):
    asyncio.create_task(schedule_remind())

async def schedule_remind():
    schedule.every().day.at("18:30").do(remind)
    while True:
        await schedule.run_pending()
        await asyncio.sleep(1)

async def remind():
    users = find_all_users_to_remind()
    for user in users:
        await bot.send_message(user[0], 'Напоминание провести ежедневные процедуры по чистке лица.')

@dp.message_handler(filters.Text('Помощь'))
@dp.message_handler(commands=['help'])
async def show_help(message: types.Message):
    await message.answer(make_help())

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer("Привет, я DermNet бот. Я помогу тебе вылечить все проблемы твоего лица!", reply_markup= greet_kb)
    user_id = message.from_user.id
    if(check_if_registrated(user_id) == 0):
        registrate_user(user_id)
        await message.answer("Регистрируем Вас как нового пользователя. Для того, чтобы подробно разобраться в функционале " + \
                             "бота, нажмите кнопку Помощь. ")
    else:
        await message.answer("Вы уже зарегистрированы в данном боте. Для того, чтобы подробно разобраться в функционале " + \
                             "бота, нажмите кнопку Помощь. ")

@dp.message_handler(commands=['classify_disease'])
@dp.message_handler(filters.Text('Классификация заболевания'))
async def classify_disease(message: types.Message):
    await PhotoForClassifyDisease.disease.set()
    await message.answer('Отправьте фото для классфификации заболевания.')

@dp.message_handler(commands=['classify_state'])
@dp.message_handler(filters.Text('Классификация состояние кожи'))
async def classify_state(message: types.Message):
    await PhotoForClassifyDisease.state.set()
    await message.answer('Отправьте фото для классфификации состояния лица.')

@dp.message_handler(commands=['detect_acne'])
@dp.message_handler(filters.Text('Обнаружение заболеваний'))
async def detect(message: types.Message):
    await PhotoForClassifyDisease.detect.set()
    await message.answer('Отправьте фото для обнаружения заболеваний лица.')

@dp.message_handler(commands=['history'])
@dp.message_handler(filters.Text('История'))
async def show_history(message):
    user_id = message.from_user.id
    message_disease_history, message_state_history, message_detect_history = prepare_history(user_id)
    await message.reply(message_disease_history)
    await message.reply(message_state_history)
    await message.reply(message_detect_history)

@dp.message_handler(content_types=['photo'], state = PhotoForClassifyDisease.disease)
async def handle_photo_disease(message):
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    user_id = message.from_user.id
    await bot.download_file(file_path, "Photos\\" + file_id + ".jpg")
    await message.reply(predict_disease(r"C:\Users\Даниил\Desktop\КТ1\pythonProject1\Photos\%s.jpg" % file_id, user_id))
    state = dp.current_state(user=message.from_user.id)
    await state.reset_state()

@dp.message_handler(content_types=['photo'], state= PhotoForClassifyDisease.state)
async def handle_photo_state(message):
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    user_id = message.from_user.id
    await bot.download_file(file_path, "Photos\\" + file_id + ".jpg")
    await message.reply(predict_state(r"C:\Users\Даниил\Desktop\КТ1\pythonProject1\Photos\%s.jpg" % file_id, user_id))
    state = dp.current_state(user=message.from_user.id)
    await state.set_state()

@dp.message_handler(content_types=['photo'], state= PhotoForClassifyDisease.detect)
async def handle_photo_detect(message):
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    user_id = message.from_user.id
    await bot.download_file(file_path, "Photos\\" + file_id + ".jpg")
    detect_photo_path, answer = make_detection(r"C:\Users\Даниил\Desktop\КТ1\pythonProject1\Photos\%s.jpg" % file_id, file_id, user_id)
    state = dp.current_state(user=message.from_user.id)
    await state.set_state()
    photo = open(detect_photo_path, 'rb')
    await bot.send_photo(chat_id=message.chat.id, photo=photo)
    await message.answer(answer)


if __name__ == '__main__':
    start_polling(dp, skip_updates=True, on_startup= on_startup)