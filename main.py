# main.py
import os
import base64
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera2 import VideoCamera2
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import argparse
import cv2 
import shutil
import random
from random import seed
from random import randint
import time
import PIL.Image
from PIL import Image, ImageChops
import numpy as np
import argparse
import imagehash
import mysql.connector
import urllib.request
import urllib.parse
from werkzeug.utils import secure_filename
from urllib.request import urlopen
import webbrowser


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="animal_reco"

)

UPLOAD_FOLDER = 'static/trained'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'abcdef'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#@app.route('/')
#def index():
#    return render_template('index.html')



@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    ff=open("msg.txt","w")
    ff.write('0')
    ff.close()

    ff=open("bc.txt","r")
    bc=ff.read()
    ff.close()
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    
        
        
    return render_template('index.html',msg=msg,bc=bc)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    
        
        
    return render_template('login.html',msg=msg)

@app.route('/login_farmer', methods=['GET', 'POST'])
def login_farmer():
    msg=""
    msg1=""
    act = request.args.get('act')
    if act=="success":
        msg1="New Farmer Register Success"
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM ani_register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            ff3=open("ulog.txt","w")
            ff3.write(uname)
            ff3.close()
            return redirect(url_for('userhome'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    
        
        
    return render_template('login_farmer.html',msg=msg,msg1=msg1)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        location=request.form['location']
        uname=request.form['uname']
        pwd=request.form['pass']
        

        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM ani_register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO ani_register(id,name,mobile,email,location,uname,pass) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,location,uname,pwd)
        mycursor.execute(sql, val)
        mydb.commit()            
        print(mycursor.rowcount, "Added Success")
        act='success'
        return redirect(url_for('login_farmer',act=act))
        
    return render_template('register.html',msg=msg)



    
@app.route('/admin', methods=['GET', 'POST'])
def admin():

    msg=""
    act="on"
    page="0"
    if request.method=='GET':
        msg = request.args.get('msg')
    if request.method=='POST':
        
        return redirect(url_for('admin2', act="on", page='0', imgg='0'))
            
    
    return render_template('admin.html', msg=msg)

@app.route('/add_data',methods=['POST','GET'])
def add_data():
    act=request.args.get("act")
    mycursor = mydb.cursor()
    if request.method == 'POST':
        
        animal = request.form['animal']
        value1 = request.form['value1']

        mycursor.execute("SELECT max(id)+1 FROM train_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO train_data(id,animal,fimg,value1) VALUES (%s, %s, %s,%s)"
        val = (maxid,animal, '',value1)
        print(sql)
        mycursor.execute(sql, val)
        mydb.commit()
        return redirect(url_for('add_photo',vid=maxid)) 

    mycursor.execute("SELECT * FROM train_data")
    data = mycursor.fetchall()

    ###
    if act=="del":
        did=request.args.get("did")

        mycursor.execute("SELECT count(*) FROM animal_img where vid=%s",(did,))
        cn = mycursor.fetchone()[0]
        if cn>0:
            mycursor.execute("SELECT * FROM animal_img where vid=%s",(did,))
            dd = mycursor.fetchall()
            for ds in dd:
                os.remove("static/frame/"+ds[2])

            mycursor.execute("delete from animal_img where vid=%s",(did,))
            mydb.commit()
                
        
        mycursor.execute("delete from train_data where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_data')) 
    ###
        
    return render_template('web/add_data.html',data=data)

@app.route('/add_photo',methods=['POST','GET'])
def add_photo():
    vid=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    #ff2=open("mask.txt","w")
    #ff2.write("face")
    #ff2.close()
    act = request.args.get('act')
    
    if request.method=='GET':
        vid = request.args.get('vid')
        ff=open("user.txt","w")
        ff.write(str(vid))
        ff.close()

    cursor = mydb.cursor()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        

        cursor.execute('delete from animal_img WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1=vid+"_"+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM animal_img")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface=vid+"_"+str(i)+".jpg"
            sql = "INSERT INTO animal_img(id, vid, animal_img) VALUES (%s, %s, %s)"
            val = (maxid, vid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update train_data set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('static/faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('view_photo',vid=vid,act='success'))
        
    
    cursor.execute("SELECT * FROM train_data")
    data = cursor.fetchall()
    return render_template('web/add_photo.html',data=data, vid=vid)

@app.route('/view_photo',methods=['POST','GET'])
def view_photo():
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM animal_img where vid=%s",(vid, ))
        value = mycursor.fetchall()


        
    return render_template('web/view_photo.html', result=value,vid=vid)


@app.route('/process_cam2',methods=['POST','GET'])
def process_cam2():
    msg=""
    ss=""
    uname=""
    act2=request.args.get("act2")
    det=""
    mess=""

    ff41=open("sms.txt","w")
    ff41.write("1")
    ff41.close()

    if request.method=='GET':
        act = request.args.get('act')
        
   
    return render_template('web/process_cam2.html',mess=mess,act=act)

@app.route('/process_cam2x',methods=['POST','GET'])
def process_cam2x():
    msg=""
    act=""
    ss=""
    value=""
    value1=""
    uname=""
    mycursor = mydb.cursor()
    act2=request.args.get("act2")
    det=""
    name=""
    mobile=""
    mess=""
    st=""
    afile=""
    sms_st=""

    ff3=open("ulog.txt","r")
    user=ff3.read()
    ff3.close()

    ff=open("bc.txt","r")
    bc=ff.read()
    ff.close()

    ff4=open("sms.txt","r")
    sms=ff4.read()
    ff4.close()

    try:
        cutoff=10
        act="1"
        mycursor = mydb.cursor()
        mycursor.execute('SELECT * FROM animal_img')
        dt = mycursor.fetchall()
        for rr in dt:
            hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
            hash1 = imagehash.average_hash(Image.open("static/faces/f1.jpg"))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            if cc1<=cutoff:
                st="1"
                vid=rr[1]
                mycursor.execute('SELECT * FROM train_data where id=%s',(vid,))
                rw = mycursor.fetchone()
                value=rw[1]
                value1=rw[3]
                msg="Animal: "+rw[1]+" detected"
                ff=open("person.txt","w")
                ff.write(msg)
                ff.close()
                print(msg)

                ##
                mycursor.execute("SELECT max(id)+1 FROM animal_detect")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1

                mycursor.execute("SELECT count(*) FROM animal_info where animal=%s",(value,))
                d1 = mycursor.fetchone()[0]
                if d1>0:
                    mycursor.execute("SELECT * FROM animal_info where animal=%s",(value,))
                    d2 = mycursor.fetchone()
                    gid=d2[0]           
                    afile="a"+str(gid)+".mp3"
                else:
                    afile="a7.mp3"

                sm=int(sms)
                
                mycursor.execute("SELECT * FROM ani_register where uname=%s",(user, ))
                row1 = mycursor.fetchone()
                mobile=str(row1[2])
                name=row1[1]
                
                mess=value+" detected"
                #url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                #webbrowser.open_new(url)
                if sm<4:
                    sms_st="1"
                sm1=sm+1
                ff41=open("sms.txt","w")
                ff41.write(str(sm1))
                ff41.close()
                    
                fn2="r"+str(maxid)+".jpg"
                shutil.copy('static/faces/f1.jpg', 'static/upload/'+fn2)
                sql = "INSERT INTO animal_detect(id,user,animal,image_name) VALUES (%s, %s,%s,%s)"
                val = (maxid,user,rw[1],fn2)
                mycursor.execute(sql, val)
                mydb.commit()
             
                break
            else:
                msg="No Animal"
                ff=open("person.txt","w")
                ff.write(msg)
                ff.close()
    except:
        print("try")
    '''ff=open("get_value.txt","r")
    value=ff.read()
    ff.close()

    if value=="":
        s=1
    else:
        st="1"
        det="Detected Animal: "+value
        mycursor.execute("SELECT max(id)+1 FROM animal_detect")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        mycursor.execute("SELECT count(*) FROM animal_info where animal=%s",(value,))
        d1 = mycursor.fetchone()[0]
        if d1>0:
            mycursor.execute("SELECT * FROM animal_info where animal=%s",(value,))
            d2 = mycursor.fetchone()
            gid=d2[0]           
            afile="a"+str(gid)+".mp3"
        else:
            afile="a7.mp3"

        if sms=="yes": 
            mycursor.execute("SELECT * FROM farmer where uname=%s",(user, ))
            row1 = mycursor.fetchone()
            mobile=row1[2]
            name=row1[1]
            
            mess=value+" detected"
            url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
            webbrowser.open_new(url)

            ff41=open("sms.txt","w")
            ff41.write("")
            ff41.close()
            
        fn2="r"+str(maxid)+".jpg"
        shutil.copy('static/trained/test.jpg', 'static/upload/'+fn2)
        sql = "INSERT INTO animal_detect(id,user,animal,image_name) VALUES (%s, %s,%s,%s)"
        val = (maxid,user,value,fn2)
        mycursor.execute(sql, val)
        mydb.commit()'''
        
   
    return render_template('web/process_cam2x.html',afile=afile,mess=mess,act=act,st=st,det=msg,mobile=mobile,name=name,sms_st=sms_st,bc=bc,value1=value1)

@app.route('/detect', methods=['GET', 'POST'])
def detect():

    ff3=open("ulog.txt","r")
    user=ff3.read()
    ff3.close()
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ani_register where uname=%s",(user, ))
    row1 = mycursor.fetchone()
    mobile=row1[2]
    name=row1[1]

    mycursor.execute("SELECT * FROM animal_detect where user=%s order by id desc",(user, ))
    data = mycursor.fetchall()

                
    return render_template('web/detect.html', data=data)


@app.route('/monitor', methods=['GET', 'POST'])
def monitor():
    msg=""

    ff=open("sms.txt","w")
    ff.write("1")
    ff.close()
            
    return render_template('monitor.html', msg=msg)


def getbox(im, color):
    bg = Image.new(im.mode, im.size, color)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    return diff.getbbox()

def split(im):
    retur = []
    emptyColor = im.getpixel((0, 0))
    box = getbox(im, emptyColor)
    width, height = im.size
    pixels = im.getdata()
    sub_start = 0
    sub_width = 0
    offset = box[1] * width
    for x in range(width):
        if pixels[x + offset] == emptyColor:
            if sub_width > 0:
                retur.append((sub_start, box[1], sub_width, box[3]))
                sub_width = 0
            sub_start = x + 1
        else:
            sub_width = x + 1
    if sub_width > 0:
        retur.append((sub_start, box[1], sub_width, box[3]))
    return retur



@app.route('/admin2', methods=['GET', 'POST'])
def admin2():
    return render_template('admin2.html', act="on", page='0', imgg='0')


###Segmentation using RNN
def crfrnn_segmenter(model_def_file, model_file, gpu_device, inputs):
    
    assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
                                        "./download_trained_model.sh").format(model_file)

    if gpu_device >= 0:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    num_images = len(inputs)
    num_channels = inputs[0].shape[2]
    assert num_channels == 3, "Unexpected channel count. A 3-channel RGB image is exptected."
    
    caffe_in = np.zeros((num_images, num_channels, _MAX_DIM, _MAX_DIM), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    start_time = time.time()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    end_time = time.time()

    print("Time taken to run the network: {:.4f} seconds".format(end_time - start_time))
    predictions = out[net.outputs[0]]

    return predictions[0].argmax(axis=0).astype(np.uint8)


def run_crfrnn(input_file, output_file, gpu_device):
    """ Runs the CRF-RNN segmentation on the given RGB image and saves the segmentation mask.
    Args:
        input_file: Input RGB image file (e.g. in JPEG format)
        output_file: Path to save the resulting segmentation in PNG format
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
    """

    input_image = 255 * caffe.io.load_image(input_file)
    input_image = resize_image(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    palette = get_palette(256)
    #PIL reads image in the form of RGB, while cv2 reads image in the form of BGR, mean_vec = [R,G,B] 
    mean_vec = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean_vec = mean_vec.reshape(1, 1, 3)

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _MAX_DIM - cur_h
    pad_w = _MAX_DIM - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Get predictions
    segmentation = crfrnn_segmenter(_MODEL_DEF_FILE, _MODEL_FILE, gpu_device, [im])
    segmentation = segmentation[0:cur_h, 0:cur_w]

    output_im = PILImage.fromarray(segmentation)
    output_im.putpalette(palette)
    output_im.save(output_file)
###Feature extraction & Classification
def DCNN_process(self):
        
        train_data_preprocess = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        test_data_preprocess = (1./255)

        train = train_data_preprocess.flow_from_directory(
                'dataset/training',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        test = train_data_preprocess.flow_from_directory(
                'dataset/test',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        ## Initialize the Convolutional Neural Net

        # Initialising the CNN
        cnn = Sequential()

        # Step 1 - Convolution
        # Step 2 - Pooling
        cnn.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        cnn.add(Flatten())

        # Step 4 - Full connection
        cnn.add(Dense(units = 128, activation = 'relu'))
        cnn.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        history = cnn.fit_generator(train,
                                 steps_per_epoch = 250,
                                 epochs = 25,
                                 validation_data = test,
                                 validation_steps = 2000)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        test_image = image.load_img('\\dataset\\', target_size=(128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        print(result)

        if result[0][0] == 1:
                print('feature extracted and classified')
        else:
                print('none')

#TCN  - Temporal Convolutional Network - Identify the Animal Intrusion
def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


    

    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs):

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def tcn_full_summary(model: Model, expand_residual_blocks=True):
        #import tensorflow as tf
        # 2.6.0-rc1, 2.5.0...
        versions = [int(v) for v in tf.__version__.split('-')[0].split('.')]
        if versions[0] <= 2 and versions[1] < 5:
            layers = model._layers.copy()  # store existing layers
            model._layers.clear()  # clear layers

            for i in range(len(layers)):
                if isinstance(layers[i], TCN):
                    for layer in layers[i]._layers:
                        if not isinstance(layer, ResidualBlock):
                            if not hasattr(layer, '__iter__'):
                                model._layers.append(layer)
                        else:
                            if expand_residual_blocks:
                                for lyr in layer._layers:
                                    if not hasattr(lyr, '__iter__'):
                                        model._layers.append(lyr)
                            else:
                                model._layers.append(layer)
                else:
                    model._layers.append(layers[i])

            model.summary()  # print summary

            # restore original layers
            model._layers.clear()
            [model._layers.append(lyr) for lyr in layers]

            

        def _build_layer(self, layer):
           
            self.layers.append(layer)
            self.layers[-1].build(self.res_output_shape)
            self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

        def build(self, input_shape):

            with K.name_scope(self.name):  # name scope used to make sure weights get unique names
                self.layers = []
                self.res_output_shape = input_shape

                for k in range(2):  # dilated conv block.
                    name = 'conv1D_{}'.format(k)
                    with K.name_scope(name):  # name scope used to make sure weights get unique names
                        conv = Conv1D(
                            filters=self.nb_filters,
                            kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate,
                            padding=self.padding,
                            name=name,
                            kernel_initializer=self.kernel_initializer
                        )
                        if self.use_weight_norm:
                            from tensorflow_addons.layers import WeightNormalization
                            # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
                            with K.name_scope('norm_{}'.format(k)):
                                conv = WeightNormalization(conv)
                        self._build_layer(conv)

                    with K.name_scope('norm_{}'.format(k)):
                        if self.use_batch_norm:
                            self._build_layer(BatchNormalization())
                        elif self.use_layer_norm:
                            self._build_layer(LayerNormalization())
                        elif self.use_weight_norm:
                            pass  # done above.

                    with K.name_scope('act_and_dropout_{}'.format(k)):
                        self._build_layer(Activation(self.activation, name='Act_Conv1D_{}'.format(k)))
                        self._build_layer(SpatialDropout1D(rate=self.dropout_rate, name='SDropout_{}'.format(k)))

                if self.nb_filters != input_shape[-1]:
                    # 1x1 conv to match the shapes (channel dimension).
                    name = 'matching_conv1D'
                    with K.name_scope(name):
                        # make and build this layer separately because it directly uses input_shape.
                        # 1x1 conv.
                        self.shape_match_conv = Conv1D(
                            filters=self.nb_filters,
                            kernel_size=1,
                            padding='same',
                            name=name,
                            kernel_initializer=self.kernel_initializer
                        )
                else:
                    name = 'matching_identity'
                    self.shape_match_conv = Lambda(lambda x: x, name=name)

                with K.name_scope(name):
                    self.shape_match_conv.build(input_shape)
                    self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

                self._build_layer(Activation(self.activation, name='Act_Conv_Blocks'))
                self.final_activation = Activation(self.activation, name='Act_Res_Block')
                self.final_activation.build(self.res_output_shape)  # probably isn't necessary

                # this is done to force Keras to add the layers in the list to self._layers
                for layer in self.layers:
                    self.__setattr__(layer.name, layer)
                self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
                self.__setattr__(self.final_activation.name, self.final_activation)

                super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

        def call(self, inputs, training=None, **kwargs):
            """
            Returns: A tuple where the first element is the residual model tensor, and the second
                     is the skip connection tensor.
            """
            
            x1 = inputs
            for layer in self.layers:
                training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
                x1 = layer(x1, training=training) if training_flag else layer(x1)
            x2 = self.shape_match_conv(inputs)
            x1_x2 = self.final_activation(layers.add([x2, x1], name='Add_Res'))
            return [x1_x2, x1]

        def compute_output_shape(self, input_shape):
            return [self.res_output_shape, self.res_output_shape]
####
                
@app.route('/training', methods=['GET', 'POST'])
def training():
    act="on"
    page="0"
    pg=""
    fn=""
    fnn=""
    imgg='1'
    tit=""
    m=0
    n=0
    
    tot=12
    
    
    #if request.method=='POST':
        
        #return redirect(url_for('training', act="on"))
    if request.method=='GET':
        act = request.args.get('act')
        page = request.args.get('page')
        imgg = request.args.get('imgg')
        n = int(page)
        if n==0:
            m = int(imgg)+1
        else:
            m = int(imgg)
            
        pg=str(n)
        page=pg
        imgg = str(m)
        
        mg = m-1
        #arr_mg=['c (1).jpeg','c (2).jpeg','g (1).jpg','g (2).jpg','']
        #rn1=randint(1,4)
        #rn2=randint(1,50)
        h=1
        while h<=12:
            if m>=1 and m<=3:
                fn='c ('+str(m)+').jpeg'
            elif m>=4 and m<=6:
                fn='e ('+str(m)+').jpg'
            elif m>=7 and m<=9:
                fn='g ('+str(m)+').jpg'
            elif m>=10 and m<=12:
                fn='h ('+str(m)+').jpeg'

            h+=1
            
        shutil.copy('static/dataset/'+fn, 'static/trained/'+fn)
        shutil.copy('static/dataset/'+fn, 'static/trained/p'+fn)
        #fn="r"+str(m)+".jpg"
        
        if m<=tot:
            act="on"
            
            if n<5:
                if n==0:
                    tit="Preprocessing"
                    image = PIL.Image.open("static/dataset/"+fn)
                    #new_image = PIL.image.resize((300, 300))
                    image.save('static/trained/'+fn)
                    
                    
                    path='static/trained/'+fn
                    im = Image.open(path)

                    pfn="p"+fn
                    path3="static/trained/"+pfn
                    for idx, box in enumerate(split(im)):
                        im.crop(box).save(path3.format(idx))
                    
                    fnn=fn
                elif n==1:
                    tit="Grayscale"
                    pfn="p"+fn
                    path3="static/trained/"+pfn
                    image = Image.open(path3).convert('L')
                    image.save(path3)
                    fnn=pfn

                   
                elif n==2:
                    tit="Segmentation"
                    pfn="p"+fn
                    img = cv2.imread('static/trained/'+pfn)
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                    # noise removal
                    kernel = np.ones((3,3),np.uint8)
                    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

                    # sure background area
                    sure_bg = cv2.dilate(opening,kernel,iterations=3)

                    # Finding sure foreground area
                    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

                    # Finding unknown region
                    sure_fg = np.uint8(sure_fg)
                    segment = cv2.subtract(sure_bg,sure_fg)
                    fname="s"+fn
                    cv2.imwrite("static/trained/"+fname, segment)
                    fnn=fname
                    
                elif n==3:
                    tit="Feature Selection"
                    image = cv2.imread("static/trained/"+fn)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    edged = cv2.Canny(gray, 50, 100)
                    image = Image.fromarray(image)
                    edged = Image.fromarray(edged)
                    fname2="f"+fn
                    edged.save("static/trained/"+fname2)

                    
                    '''image = cv2.imread("static/trained/"+fn)
                    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #edged = cv2.Canny(gray, 50, 100)
                    fname2="p"+fn
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

                    #canny
                    img_canny = cv2.Canny(image,50,100)
                    
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
      
                    # Calcution of Sobelx 
                    sobelx = cv2.Sobel(img_gaussian,cv2.CV_64F,1,0,ksize=5) 
                      
                    # Calculation of Sobely 
                    sobely = cv2.Sobel(img_gaussian,cv2.CV_64F,0,1,ksize=5) 
                      
                    # Calculation of Laplacian 
                    laplacian = cv2.Laplacian(image,cv2.CV_64F)
                    
                    cv2.imwrite("static/trained/"+fname2, img_canny)'''
                    fnn=fname2
                elif n==4:
                    x=1
                    tit="Classification"
                    
                    fnn=fn
                else:
                    
                    tit="Classified"
                    fnn=fn
                n = int(page)+1
                pg=str(n)
                page=pg
            else:
                tit="Classified"
                fnn=fn
                page='0'
                if m==tot:
                    
                    act="ok"
               
        else:
            tit="Classified File Created"
            act="ok"
                
    #return send_file(path, as_attachment=True)
    return render_template('training.html',tit=tit, imgg=imgg, page=page, act=act, fn=fnn)

@app.route('/test_upload1', methods=['GET', 'POST'])
def test_upload1():
    if request.method=='POST':
        #print("d")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        file_type = file.content_type
        # if user does not select file, browser also
        # submit an empty part without filename
        tf=file.filename
        ff=open("log.txt","w")
        ff.write(tf)
        ff.close()
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = "m1.jpg"
            filename = secure_filename(fname)
                
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('test', act="on", page='0', imgg='0'))
    return render_template('test_upload1.html')
    

@app.route('/test_upload', methods=['GET', 'POST'])
def test_upload():
    if request.method=='POST':
        #print("d")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        file_type = file.content_type
        # if user does not select file, browser also
        # submit an empty part without filename
        tf=file.filename
        ff=open("log.txt","w")
        ff.write(tf)
        ff.close()
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = "m1.jpg"
            filename = secure_filename(fname)
                
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('test', act="on", page='0', imgg='0'))
    return render_template('test_upload.html')


#@app.route('/test', methods=['GET', 'POST'])
#def test():
#    return render_template('test.html', act="on", page='0', imgg='0')

@app.route('/testing', methods=['GET', 'POST'])
def testing():
    act="on"
    page="0"
    pg=""
    fn=""
    fnn=""
    imgg='1'
    tit=""
    m=0
    n=0
    
    tot=1
    
    
    #if request.method=='POST':
        
        #return redirect(url_for('training', act="on"))
    if request.method=='GET':
        act = request.args.get('act')
        page = request.args.get('page')
        imgg = request.args.get('imgg')
        n = int(page)
        if n==0:
            m = int(imgg)+1
        else:
            m = int(imgg)
            
        pg=str(n)
        page=pg
        imgg = str(m)
        
        mg = m-1
        
        fn="m1.jpg"
        
        if m<=tot:
            act="on"
            
            if n<5:
                if n==0:
                    tit="Preprocessing"
                    image = PIL.Image.open("static/trained/"+fn)
                    #new_image = PIL.image.resize((300, 300))
                    image.save('static/trained/'+fn)
                    
                    
                    path='static/trained/'+fn
                    im = Image.open(path)

                    pfn="q"+fn
                    path3="static/trained/"+pfn
                    for idx, box in enumerate(split(im)):
                        im.crop(box).save(path3.format(idx))
                    
                    fnn=fn
                elif n==1:
                    tit="Grayscale"
                    pfn="q"+fn
                    path3="static/trained/"+pfn
                    image = Image.open(path3).convert('L')
                    image.save(path3)
                    fnn=pfn

                   
                elif n==2:
                    tit="Segmentation"
                    pfn="q"+fn
                    img = cv2.imread('static/trained/'+pfn)
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                    # noise removal
                    kernel = np.ones((3,3),np.uint8)
                    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

                    # sure background area
                    sure_bg = cv2.dilate(opening,kernel,iterations=3)

                    # Finding sure foreground area
                    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

                    # Finding unknown region
                    sure_fg = np.uint8(sure_fg)
                    segment = cv2.subtract(sure_bg,sure_fg)
                    fname="s"+fn
                    cv2.imwrite("static/trained/"+fname, segment)
                    fnn=fname
                    
                elif n==3:
                    tit="Feature Selection"
                    image = cv2.imread("static/trained/"+fn)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    edged = cv2.Canny(gray, 50, 100)
                    fname2="f"+fn
                    cv2.imwrite("static/trained/"+fname2, edged)
                    fnn=fname2
                elif n==4:
                    x=1
                    tit="Compare with Classified File"
                    
                    fnn=fn
                    return redirect(url_for('result'))
                else:
                    
                    tit="Classified"
                    fnn=fn
                n = int(page)+1
                pg=str(n)
                page=pg
            else:
                tit="Classified"
                fnn=fn
                page='0'
                if m==tot:
                    
                    act="ok"
               
        else:
            act="ok"
                
    #return send_file(path, as_attachment=True)
    return render_template('testing.html',tit=tit, imgg=imgg, page=page, act=act, fn=fnn)



@app.route('/anitest', methods=['GET', 'POST'])
def anitest():
    msg=""
    act=""
    value="z"
    mess=""
    mobile=""
    name=""
    st=""

    ff=open("msg.txt","r")
    mc=ff.read()
    ff.close()

    ff=open("bc.txt","r")
    bc=ff.read()
    ff.close()
    
    mcount=int(mc)
    mcc=mcount+1
    
    animal=""
    xn=randint(1, 65)
    an=randint(1, 100)
    an1=str(an)
    fnn="r1.jpg"
    print("animal-----")
    print(an1)
    act=str(xn)
    #str(xn)

    if an>=1 and an<=5:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Elephant"
        msg=animal+" Detected"
        value='a'
    elif an>=6 and an<=10:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Monkey"
        msg=animal+" Detected"
        value='b'
    elif an>=11 and an<=15:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Lion"
        msg=animal+" Detected"
        value='c'
    elif an>=16 and an<=20:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Tiger"
        msg=animal+" Detected"
        value='d'
    elif an>=21 and an<=25:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Cheta"
        msg=animal+" Detected"
        value='e'
    elif an>=26 and an<=30:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Panda"
        msg=animal+" Detected"
        value='f'
    elif an>=31 and an<=35:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Fox"
        msg=animal+" Detected"
        value='g'
    elif an>=36 and an<=40:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Hyena"
        msg=animal+" Detected"
        value='h'
    elif an>=41 and an<=45:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Bison"
        msg=animal+" Detected"
        value='i'
    elif an>=46 and an<=50:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Leoprd"
        msg=animal+" Detected"
        value='j'
    elif an>=51 and an<=55:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Girafee"
        msg=animal+" Detected"
        value='k'
    elif an>=56 and an<=60:
        act="1"
        fnn="r"+an1+".jpg"
        animal="Pig"
        msg=animal+" Detected"
        value='l'
    elif an>=61 and an<=65:
        act="1"
        fnn="r"+an1+".jpg"
        animal="ostrich"
        msg=animal+" Detected"
        value='m'
    else:
        act=""
        animal=""
        msg="No Animals"
        value='z'
        
    print("value="+value+"----"+animal)

    ff=open("sms.txt","r")
    sms=ff.read()
    ff.close()
            
    if act=="1":    
        if mcount<2:
            ff=open("msg.txt","w")
            ff.write(str(mcc))
            ff.close()

            ssm=int(sms)
            
            if ssm<5:
                st="1"
            ssm+=1
            ssm1=str(ssm)
            ff=open("sms.txt","w")
            ff.write(ssm1)
            ff.close()

            cursor = mydb.cursor()
            cursor.execute('SELECT * FROM admin')
            account = cursor.fetchone()
            mobile=str(account[2])
            #url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name=Farmer&mess="+msg+"&mobile="+str(mobile)
            #webbrowser.open_new(url)
    '''if xn<=7:
        fnn="r"+str(xn)+".jpg"
    
    if act=="1":
        animal="Cow"
        msg="Cow Detected"
    elif act=="2":
        animal="Cow"
        msg="Cow Detected"
    elif act=="3":
        animal="Elephant"
        msg="Elephant Detected"
    elif act=="4":
        animal="Elephant"
        msg="Elephant Detected"
    elif act=="5":
        animal="Goat"
        msg="Goat Detected"
    elif act=="6":
        animal="Goat"
        msg="Goat Detected"
    elif act=="7":
        animal="Goat"
        msg="Goat Detected"
    else:
        animal=""
        msg="No Animals"'''
    
    
    if animal=="":
        print("")
    else:
        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM ani_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO ani_data(id,animal) VALUES (%s, %s)"
        val = (maxid,animal)
        mycursor.execute(sql, val)
        mydb.commit()    
    ##################
    # construct the argument parse 
    parser = argparse.ArgumentParser(
        description='Script to run MobileNet-SSD object detection network ')
    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                      help='Path to text network file: '
                                           'MobileNetSSD_deploy.prototxt for Caffe model or '
                                           )
    parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                     help='Path to weights: '
                                          'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                          )
    parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
    args = parser.parse_args()

    # Labels of Network.
    classNames = { 0: 'background',
        1: 'mobile', 2: 'bicycle', 3: 'cup', 4: 'glass',
        5: 'bottle', 6: 'paper', 7: 'car', 8: 'cat', 9: 'chair',
        10: 'cow', 11: 'diningtable', 12: 'goat', 13: 'horse',
        14: 'motorbike', 15: 'person', 16: 'goat',
        17: 'elephant', 18: 'cow', 19: 'cellphone', 20: 'tvmonitor' }

    # Open video file or capture device. 
    '''if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)'''

    #Load the Caffe model 
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

    #while True:
    # Capture frame-by-frame
    #ret, frame = cap.read()
    frame = cv2.imread("static/animals/"+fnn)
    frame_resized = cv2.resize(frame,(200,200)) # resize frame for prediction

    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size. 
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300)
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob 
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()

    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > args.thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0 
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)
            # Draw location of object  
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))
            try:
                y=yLeftBottom
                h=yRightTop-y
                x=xLeftBottom
                w=xRightTop-x
                image = cv2.imread("static/animals/"+fnn)
                mm=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imwrite("static/result/"+fnn, mm)
                cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                gg="segment.jpg"
                cv2.imwrite("static/result/"+gg, cropped)
                #mm2 = PIL.Image.open('static/trained/'+gg)
                #rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                #rz.save('static/trained/'+gg)
            except:
                print("none")
                #shutil.copy('getimg.jpg', 'static/trained/test.jpg')
            # Draw label and confidence of prediction in frame resized
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                print(label) #print class and confidence
    ####################
    return render_template('anitest.html',act=act,msg=msg,fnn=fnn,value=value,bc=bc,mess=mess,mobile=mobile,st=st)

@app.route('/test', methods=['GET', 'POST'])
def test():

    return render_template('test.html')
    
@app.route('/result', methods=['GET', 'POST'])
def result():
    res=""
    afile="a3.mp3"
    password_provided = "xyz" # This is input in the form of a string
    password = password_provided.encode() # Convert to type bytes
    salt = b'salt_' # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password)) # Can only use kdf once
    f2=open("log.txt","r")
    vv=f2.read()
    f2.close()
    vv1=vv.split('.')
    tff3=vv1[0]
    tff4=tff3[1:]
    rid=int(tff4)
    input_file = 'test.encrypted'
    with open(input_file, 'rb') as f:
        data = f.read()

    fernet = Fernet(key)
    encrypted = fernet.decrypt(data)
    value=encrypted.decode("utf-8")
    dar=value.split('|')
    rr=rid-1
    dv=dar[rr]
    drw=dv.split('-')
    v=drw[1]
    '''if v=="a1.flac":
        lf="Cow"
    elif v=="a2.mp3":
        lf="Elephant"
    else:
        lf="Goat"'''
    
    return render_template('result.html',res=lf,afile=v)


@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ani_data order by id desc")
    data = mycursor.fetchall()
    return render_template('userhome.html',data=data)

@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))

###################
def gen2(camera2):
    
    while True:
        frame = camera2.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed2')
        

def video_feed2():
    return Response(gen2(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
