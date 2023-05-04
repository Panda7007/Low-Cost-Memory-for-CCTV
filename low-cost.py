#Untuk capture dan deteksi
import cv2
import numpy as np
import time
import datetime as dt


#Mengambil gambar dari sumber
cap = cv2.VideoCapture(0) 

#Menggambil dataset onnx untuk di proses
net = cv2.dnn.readNetFromONNX(r"crowddetector.onnx")
#Buat menandai class berdasarkan urutan class di roboflow
classes = ["Orang : "]


tanggal = dt.datetime.now().date()
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(str(tanggal)+'Lowcost--5fps-2.avi', fourcc, 20.0, (640, 480))


#settingan untuk melihat FPS
starting_time = time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_COMPLEX
   


while True:
    img = cap.read()[1]
    frm = img
    frame_id += 1
    if img is None:
        break
    img = cv2.resize(img, (500,300))
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255, size=(640,640), mean=[0, 0, 0], swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]



    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence >= 0.25:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] >= 0.25:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)
                
                # Write the frame to the video file
                out.write(frm)
    

    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.25,0.25)
    cv2.putText(img,"Jumlah Orang : " + str(len(indices)), (10,90), font, 0.4,(200,0,0),2)
    
    waktu = time.asctime()
    jam = dt.datetime.now().hour
    tanggal = dt.datetime.now().date()
    
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    

    cv2.putText(img, "Fps : " + str(fps), (10,30), font, 0.3, (255,0,0),1)
    cv2.putText(img, "Waktu : "+ str(waktu), (10, 50), font, 0.3, (200,0,0),1)
    cv2.putText(img, "Jam : "+ str(jam), (10, 70), font, 0.3, (200,0,0),1)
    cv2.putText(img, "Tekan Q untuk keluar & Save ", (10, 280), font, 0.3, (200,200,200),1)
 
    
    #Boxing object yang terdetksi
    for i in indices:
        x1,y1,w,h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),2)
        cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.3,(0,0,255),2)
    
    cv2.imshow("VIDEO",img)
    
      
    #Deteksi manusia untuk menjalankan alaram
    
    
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

# Release the capture and writer objects
cap.release()
out.release()

# Destroy all windows
cv2.destroyAllWindows()


