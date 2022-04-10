import torch
from aiogram import md, types
import datetime
from database_process import write_disease, write_state, write_detected, allow_to_write, get_disease_story, get_state_story, get_detected_story

dictionary_of_disease = {
    0 : 'Акне', 1 : 'Актиническим кератозом', 2 : 'Дерматитом', 3: 'Пигментным пятноv',
    4: 'Кандидозом кожи', 5: 'Сосудистыми опухолями', 6: 'Вирусной инфекцией'
}

dictionary_of_state = {0: "Легкая пораженность", 1: "Средняя пораженность", 2: "Тяжелая пораженность"}

def make_help():
    return  md.text(
        md.text("Предсказания данного бота не являются действительными врачебными диагнозами!"),
        md.text("\n"),
        md.text("Это бот имеет несколько фнкций:"),
        md.text("   🔸Для классификации поражения нажмите кнопку 'Классификация заболевания' или введите /classify_disease"),
        md.text("   🔸Для классификации состояния лица нажмите кнопку 'Классификация состояния' или введите /classify_state"),
        md.text("   🔸Для обнаружения поражения нажмите кнопку 'Классификация заболевания' или введите /detect_acne"),
        md.text("   🔸Для показа истории заболеваний нажмите кнопку 'История' или введите /history"),
        md.text("\n"),
        md.text("После применения первых трех комманд выведется запрос на отправку фото, по которому будет произведен анализ."),
        sep='\n',
    )

def make_text_prediction(prediction: torch.tensor, user_id):
    prediction = prediction.squeeze()
    write_disease(user_id, torch.argmax(prediction).item())
    return md.text(
        md.text('Модель предсказала вероятности, что поражение является:'),
        md.text('🔸', 'Акне:', round(prediction[0].item(), 2)),
        md.text('🔸', 'Актинический кератоз:', round(prediction[1].item(), 2)),
        md.text('🔸', 'Дерматит:', round(prediction[2].item(), 2)),
        md.text('🔸', 'Пигментное пятно:', round(prediction[3].item(), 2)),
        md.text('🔸', 'Кандидоз кожи:', round(prediction[4].item(), 2)),
        md.text('🔸', 'Сосудистые опухоли:', round(prediction[5].item(), 2)),
        md.text('🔸', 'Вирусная инфекция:', round(prediction[6].item(), 2)),
        md.text("\n"),
        md.text("Наиболее вероятно болезнь на изображении: " + dictionary_of_disease[torch.argmax(prediction).item()]),
        sep='\n',
    )

def make_text_state_prediction(prediction: torch.tensor, user_id):
    write_state(user_id, prediction)
    match prediction:
        case 0:
            return md.text(
                md.text('Модель предсказала, что состояние вашего лица: легкое пораженность'),
                md.text('Даже у людей с красивой здоровой кожей иногда появляются прыщики на лице.'),
                md.text('Это связано с незначительными гормональными изменениями, иногда — с механическими повреждениями кожи.'),
                md.text('Обычно такие прыщи проходят сами и никаких следов не оставляют.'),
                md.text(' Эта степень тяжести акне еще не требует медикаментозного лечения, достаточно правильных гигиенических процедур.'),
                sep = '\n',
            )
        case 1:
            return md.text(
                md.text('Модель предсказала, что состояние вашего лица: среднее пораженность'),
                md.text('Как правило, при средней степени узлов еще нет, но кожа вокруг прыщей и комедонов воспаленная, приобретает неприятный на вид синюшно-розовый цвет, доставляя больному дискомфорт, особенно при общении с людьми.'),
                md.text('Самостоятельно пациенту справиться с акне на этой стадии уже не получается и он обращается за консультацией к дерматологу.'),
                md.text('При средней степени тяжести акне вряд ли получится обойтись только наружными средствами для лечения воспаленных прыщей, необходимо назначение комплексной терапии.'),
                sep='\n',
            )
        case 2:
            return md.text(
                md.text('Модель предсказала, что состояние вашего лица: тяжелая пораженность'),
                md.text('Тяжелая степень акне характеризуется появлением больших воспаленных участков кожи, болезненностью узлов.'),
                md.text('Акне тяжелой степени лечить только наружными средствами бесполезно. Требуется системная медикаментозная терапия, в состав которой обязательно входят антибиотики.'),
                md.text('Антибиотиками первого выбора при акне являются препараты тетрациклинового ряда.'),
                sep='\n',
            )

def detection_prediction_text(draw_boxes, user_id):
    count_detected_disease = len(draw_boxes)
    write_detected(user_id, count_detected_disease)
    return "Модель нашла на вашем лице %s заболеваний, которые требуют лечения. " % len(draw_boxes) + \
    "Для эффективного лечения сделайте детальное фото, используйте функцию классификации и получите тип заболевания."

def prepare_history(user_id):
    if (allow_to_write(user_id) == 1):
        disease_story = get_disease_story(user_id)
        format = "%Y-%m-%d %H:%M:%S.%f"
        message_disease_history = "Последние 5 диагностированных у Вас заболевания: \n"
        for i in disease_story:
            disease = i[0]
            request_date = datetime.datetime.strptime(i[1], format)
            request_date = request_date.replace(second=0, microsecond=0)
            message_disease_history += "🔸%s-%s-%s %s:%s - %s\n" % (request_date.day, request_date.month, request_date.year, request_date.hour, request_date.minute, disease)

        state_story = get_state_story(user_id)
        message_state_history = "Последние пять диагностированных состояний кожи: \n"
        for i in state_story:
            state = i[0]
            request_date = datetime.datetime.strptime(i[1], format)
            request_date = request_date.replace(second=0, microsecond=0)
            message_state_history += "🔸%s-%s-%s %s:%s - %s\n" % (request_date.day, request_date.month, request_date.year, request_date.hour, request_date.minute, dictionary_of_state[state])

        detected_story = get_detected_story(user_id)
        message_detect_history = "Последние пять результатов обнаружений заболеваний: \n"
        for i in detected_story:
            count = i[0]
            request_date = datetime.datetime.strptime(i[1], format)
            request_date = request_date.replace(second=0, microsecond=0)
            message_detect_history += "🔸%s-%s-%s %s:%s - %s\n" % (request_date.day, request_date.month, request_date.year, request_date.hour, request_date.minute, count)
        return message_disease_history, message_state_history, message_detect_history
