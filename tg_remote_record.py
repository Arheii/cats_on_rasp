from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import telegram
import os
import cv2 #used only for save photo and send from tg. can be refactor
from picamera import PiCamera
from time import sleep
from datetime import datetime
from uznavalka import ClassificatorNet
from yolo_detection import YoloNet



TOKEN = os.environ['TG_TOKEN'] # your telegram bot token
WHITE_LIST_IDS = [358201765, 253672630, 53035836, 1049316533] #= os.environ['TG_CHAT_ID'] # your ids_chat with bot (update.bot.chat_id)

ch_start_record = b'\xF0\x9F\x94\xB4'.decode("utf-8", "strict")
ch_stop_record = b'\xF0\x9F\x94\xB3'.decode("utf-8", "strict")
ch_photo_camera = b'\xF0\x9F\x93\xB7'.decode("utf-8", "strict")
ch_warmup = b'\xF0\x9F\x94\xA5'.decode("utf-8", "strict")
ch_net = b'\xF0\x9F\x8E\xAF'.decode("utf-8", "strict")
ch_off = b'\xF0\x9F\x93\xB4'.decode("utf-8", "strict")
yolo_medium = b"\xF0\x9F\x90\xA2".decode("utf-8", "strict") + 'Yolo medium'
yolo_tiny= b"\xF0\x9F\x90\x8A".decode("utf-8", "strict") + 'Yolo tiny'
resnet50 = b"\xF0\x9F\x90\xA3".decode("utf-8", "strict") + 'ResNet50'
main_menu = "Menu"
off_net = b"\xF0\x9F\x9A\xB7".decode("utf-8", "strict")

#States
#custom_keyboard = [[ch_start_record, ch_stop_record, ch_photo_camera], [ch_warmup, ch_net, ch_off]]
kb_state_idle = [[ch_warmup]]
kb_state_active_not_recording = [[ch_start_record, ch_photo_camera], [ch_net, ch_off]]
kb_state_active_is_recording = [[ch_stop_record, ch_photo_camera], [ch_net, ch_off]]
kb_state_choise_net = [[yolo_tiny, yolo_medium, resnet50], [off_net, main_menu]]


class BotRecorder():
    def __init__(self):
        self.camera = None
        self.net_rec = None
        self.doing_record = False
        self.choise_net = False
        self.current_net = None
        self.path_img ='/home/pi/sv/datasets/img_from_pi/'
        self.path_video = '/home/pi/sv/datasets/video_from_pi/'
        os.makedirs(self.path_img, exist_ok=True)
        os.makedirs(self.path_video, exist_ok=True)

    def _kb(self):
        if self.choise_net:
            kb = kb_state_choise_net
        elif self.camera is None:
            kb = kb_state_idle
        else:
            if self.doing_record:
                kb = kb_state_active_is_recording
            else:
                kb = kb_state_active_not_recording
        
        return telegram.ReplyKeyboardMarkup(kb, resize_keyboard=True)

    def warmup(self, update, context):
        self.camera = PiCamera()
        text = "Camera is ready!"
        context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=self._kb())


    def start(self, update, context):
        if self.camera is None:
            text="First warmup camera"
        elif self.doing_record:
            text="Camera busy"
        else:
            f_name = self.path_video + str_now() + '.h264'
            self.doing_record = True
            self.camera.start_recording(f_name)
            text = 'Starting record'
        context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=self._kb())


    def stop(self, update, context):
        if self.camera is None:
            text = 'Firstly warmup camera, then start record'
        elif not self.doing_record:
            text = "Nothing recording"
        else:
            self.camera.stop_recording()
            self.doing_record = False
            text = "Video recording stop"
        context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=self._kb())
       

    def shot(self, update, context):
        if self.camera is None:
            text = "Firstly warmup camera"
        elif self.doing_record:
            text = "Camera busy. Stop the recording first."
        else:
            f_name = self.path_img + str_now() + '.jpg'
            self.camera.capture(f_name)
            context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(f_name, 'rb'))
            text = "Shot saved"
            if self.net_rec is not None:
                start_time = datetime.now()
                if self.current_net == 'resnet50':
                    self.net_rec.recognition(file=f_name)
                    pred_time = datetime.now() - start_time
                    text_2 = f"All predictions: {self.net_rec.percents}, time_spend: {pred_time}"
                    context.bot.send_message(chat_id=update.effective_chat.id, text=text_2)
                elif self.current_net == 'yolo_tiny' or self.current_net == 'yolo_medium':
                    self.net_rec.recognition(file=f_name)
                    frame = self.net_rec.frame_with_inf()
                    f_name_inf = f_name[:-4] + '_inf.png'
                    print(f_name_inf)
                    cv2.imwrite(f_name_inf, frame)

                    pred_time = datetime.now() - start_time
                    cat = f'net_rec.cat={self.net_rec.cat}\n' if self.net_rec.cat else ''
                    raw_cats = f'rawcat={self.net_rec.cat_raw}\n' if self.net_rec.cat_raw else ''
                    person = f'net_rec.person={self.net_rec.person}\n' if self.net_rec.person else ''
                    raw_person = f'rawperson={self.net_rec.person_raw}\n' if self.net_rec.person_raw else ''
                    text_2 = f"time_spend: {pred_time}\n{cat}{raw_cats}{person}{raw_person}"

                    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(f_name_inf, 'rb'))
                    context.bot.send_message(chat_id=update.effective_chat.id, text=text_2)
        context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=self._kb())


    def net(self, update, context):
        self.choise_net = True
        if self.net_rec is None:
            text = 'Not used any net. You can choise one of them and then will do shot'
        else:
            text = f'Used {self.current_net}'
        context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=self._kb())

    def load_net(self, update, context, net):
        start_time = datetime.now()
        if net == 'yolo_tiny': #and self.current_net is not 'yolo_tiny':
            self.net_rec = YoloNet(vers='v3_tiny')
            self.current_net = 'yolo_tiny'
            text = f'Yolo tiny loaded for {datetime.now() - start_time}'
        elif net == 'yolo_medium':
            self.net_rec = YoloNet(vers='v3_medium')
            self.current_net = 'yolo_medium'
            text = f'Yolo medium loaded for {datetime.now() - start_time}'
        elif net == 'resnet50':
            self.net_rec = ClassificatorNet()
            self.current_net = 'resnet50'
            text = f'resnet50 loaded for {datetime.now() - start_time}'
        elif net == 'off_net':
            self.net_rec = None
            self.current_net = False
            text = "Classification net Off"
        context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=self._kb())

    def ret_to_menu(self, update, context):
        self.choise_net = False
        text = 'returning to main menu'
        context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=self._kb())

    def off(self, update, context):
        self.camera.close()
        self.camera = None
        context.bot.send_message(chat_id=update.effective_chat.id, text="camera off", reply_markup=self._kb())

    def message(self, update, context):
        user_id = update.effective_user.id
        if user_id not in WHITE_LIST_IDS:
            print("Unauthorized access denied for {}.".format(user_id))
            return
        msg = update.message.text
        print("Received message:"+msg)
        if msg == ch_warmup:
            self.warmup(update,context)
        elif msg == ch_start_record:
            self.start(update, context)
        elif msg == ch_stop_record:
            self.stop(update, context)
        elif msg == ch_photo_camera:
            self.shot(update, context)
        elif msg == ch_net:
            self.net(update, context)
        elif msg == ch_off:
            self.off(update, context)
        elif msg == resnet50:
            self.load_net(update, context, 'resnet50')
        elif msg == yolo_medium:
            self.load_net(update, context, 'yolo_medium')
        elif msg == yolo_tiny:
            self.load_net(update, context, 'yolo_tiny')
        elif msg == off_net:
            self.load_net(update, context, 'off_net')
        elif msg == main_menu:
            self.ret_to_menu(update, context)
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, dont't know what to do with '%s'" % msg, reply_markup=self._kb())


def str_now():
    return datetime.now().strftime("%d-%m-%Y-%H-%M-%S")


if __name__ == '__main__':
    try:
        updater = Updater(token=TOKEN, use_context=True)
        dp = updater.dispatcher
        bt = BotRecorder()

        message_handler = MessageHandler(Filters.text, bt.message)
        dp.add_handler(message_handler)

        updater.start_polling()

    finally:
        updater.idle()
        if bt.camera is not None:
            bt.camera.close()
