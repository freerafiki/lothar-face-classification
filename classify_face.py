# conda activate /Users/Palma/Documents/Opencampus/conda_envs/oc_env
import pdb
import face_recognition
image = face_recognition.load_image_file("photos/IMG-20160725-WA0009.jpg")
face_locations = face_recognition.face_locations(image)

known_image = face_recognition.load_image_file("photos/IMG-20160801-WA0006.jpg")
unknown_image = face_recognition.load_image_file("photos/IMG-20160808-WA0000.jpg")

known_image_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
enc_image = face_recognition.face_encodings(image)[0]

results = face_recognition.compare_faces([known_image_encoding, enc_image], unknown_encoding)
pdb.set_trace()
