import cv2
import numpy as np
import os
import pandas as pd
import pytesseract
import re
import textdistance
from datetime import date
from operator import itemgetter

ROOT_PATH = os.getcwd()
# IMAGE_PATH = os.path.join(ROOT_PATH, 'Kywa.jpg')
LINE_REC_PATH = os.path.join(ROOT_PATH, 'data/ID_CARD_KEYWORDS.csv')
CITIES_REC_PATH = os.path.join(ROOT_PATH, 'data/KOTA.csv')
RELIGION_REC_PATH = os.path.join(ROOT_PATH, 'data/RELIGIONS.csv')
MARRIAGE_REC_PATH = os.path.join(ROOT_PATH, 'data/MARRIAGE_STATUS.csv')
JENIS_KELAMIN_REC_PATH = os.path.join(ROOT_PATH, 'data/JENIS_KELAMIN.csv')
PROVINCE_REC_PATH = os.path.join(ROOT_PATH, 'data/PROVINSI.csv')
DISTRIC_REC_PATH = os.path.join(ROOT_PATH, 'data/KECAMATAN.csv')
PEKERJAAN_REC_PATH = os.path.join(ROOT_PATH, 'data/PEKERJAAN.csv')
NEED_COLON = [3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]
NEXT_LINE = 9
ID_NUMBER = 3

def convertScale(img, alpha, beta):
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    # auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

# ------------------------------------------------------------------------------

def ocr_raw(image):
    # img_raw = cv2.imread(image_path)
    # image = automatic_brightness_and_contrast(image)

    image = cv2.resize(image, (50 * 16, 500))
    # cv2.imshow("test1", image)

    image = automatic_brightness_and_contrast(image)
    # cv2.imshow("test2", image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # img_gray = cv2.equalizeHist(img_gray)
    # img_gray = cv2.fastNlMeansDenoising(img_gray, None, 3, 7, 21)
    id_number = return_id_number(image, img_gray)
    cv2.fillPoly(img_gray, pts=[np.asarray([(540, 150), (540, 499), (798, 499), (798, 150)])], color=(255, 255, 255))
    th, threshed = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    result_raw = pytesseract.image_to_string(threshed, lang="ind")

    return result_raw, id_number

def strip_op(result_raw):
    result_list = result_raw.split('\n')
    new_result_list = []

    for tmp_result in result_list:
        if tmp_result.strip(' '):
            new_result_list.append(tmp_result)

    return new_result_list

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes

def return_id_number(image, img_gray):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, rectKernel)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)

    threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    cur_img = image.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    copy = image.copy()

    locs = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)

        # ar = w / float(h)
        # if ar > 3:
        # if (w > 40 ) and (h > 10 and h < 20):
        if h > 10 and w > 100 and x < 300:
            img = cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            locs.append((x, y, w, h, w * h))

    locs = sorted(locs, key=itemgetter(1), reverse=False)

    # nik = image[locs[1][1] - 15:locs[1][1] + locs[1][3] + 15, locs[1][0] - 15:locs[1][0] + locs[1][2] + 15]
    # text = image[locs[2][1] - 10:locs[2][1] + locs[2][3] + 10, locs[2][0] - 10:locs[2][0] + locs[2][2] + 10]

    check_nik = False

    try:
        nik = image[locs[1][1] - 15:locs[1][1] + locs[1][3] + 15, locs[1][0] - 15:locs[1][0] + locs[1][2] + 15]
        check_nik = True
    except Exception as e:
        print(e)

    if check_nik == True:
        img_mod = cv2.imread("data/module2.png")

        ref = cv2.cvtColor(img_mod, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(ref, 66, 255, cv2.THRESH_BINARY_INV)[1]

        refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refCnts = sort_contours(refCnts, method="left-to-right")[0]

        digits = {}
        for (i, c) in enumerate(refCnts):
            (x, y, w, h) = cv2.boundingRect(c)
            roi = ref[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            digits[i] = roi

        gray_nik = cv2.cvtColor(nik, cv2.COLOR_BGR2GRAY)
        group = cv2.threshold(gray_nik, 127, 255, cv2.THRESH_BINARY_INV)[1]

        digitCnts, hierarchy_nik = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nik_r = nik.copy()
        cv2.drawContours(nik_r, digitCnts, -1, (0, 0, 255), 3)

        gX = locs[1][0]
        gY = locs[1][1]
        gW = locs[1][2]
        gH = locs[1][3]

        ctx = sort_contours(digitCnts, method="left-to-right")[0]

        locs_x = []
        for (i, c) in enumerate(ctx):
            (x, y, w, h) = cv2.boundingRect(c)
            if h > 10 and w > 10:
                img = cv2.rectangle(nik_r, (x, y), (x + w, y + h), (0, 255, 0), 2)
                locs_x.append((x, y, w, h))


        output = []
        groupOutput = []

        for c in locs_x:
            (x, y, w, h) = c
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))

            scores = []
            for (digit, digitROI) in digits.items():
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            groupOutput.append(str(np.argmax(scores)))

        cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        output.extend(groupOutput)
        return ''.join(output)
    else:
        return ""

def parsing(data):
    parsed_data = {
        'PROVINSI' : '-',
        'KOTA/KAB' : '-',
        'NIK': '-',
        'Nama': '-',
        'Tempat/Tgl_Lahir': '-',
        'Jenis_Kelamin': '-',
        'Gol_Darah' : '-',
        'Alamat' : '-',
        'Kel/Desa' : '-',
        'Kecamatan' : '-',
        'Agama' : '-',
        'RT/RW' : '-',
        'Status_Perkawinan': '-',
        'Pekerjaan': '-',
        'Kewarganegaraan': '-',
        'Berlaku_Hingga': '-'
    }
        
    for index, entry in enumerate(data):
        key = entry[0]
        value = ' '.join(entry[2:]).replace(':', '').strip()
        if key == 'PROVINSI':
            parsed_data['PROVINSI'] = ' '.join(entry[1:]).replace(':', '').strip()
        if key == 'KOTA' or key == 'KABUPATEN':
            parsed_data['KOTA/KAB'] = ' '.join(entry[1:]).replace(':', '').strip()
        if key == 'NIK':
            parsed_data['NIK'] = value
        if key == 'Nama':
            parsed_data['Nama'] = ''.join([char for char in value if not char.isdigit()])
        if key == 'Tempat/Tgl' or key == 'Lahir':
            parsed_data['Tempat/Tgl_Lahir'] = value
        if key == 'Jenis' or key == 'Kelamin':
            removedigit(value)
            if 'LAKI' in value or 'LAKI-LAKI' in value:
                parsed_data['Jenis_Kelamin'] = 'LAKI-LAKI'
            elif 'PEREMPUAN' in value:
                parsed_data['Jenis_Kelamin'] = 'PEREMPUAN'
            else:
                parsed_data['Jenis_Kelamin'] = '-'
                                    
        if key == 'Gol.' or key == 'Darah':
            removedigit(value)
            if value.find("A") != -1:
                parsed_data['Gol_Darah'] = 'A'
            elif value.find("B") != -1:
                parsed_data['Gol_Darah'] = 'B'
            elif value.find("AB") != -1:
                parsed_data['Gol_Darah'] = 'AB'
            elif value.find("O") != -1:
                parsed_data['Gol_Darah'] = 'O'
            else :
                parsed_data['Gol_Darah'] = '-'  
                  
        if key == 'Alamat':
            parsed_data['Alamat'] = value
        if key == 'RT/RW':
            parsed_data['RT/RW'] = value
        if key == 'Kel/Desa':
            parsed_data['Kel/Desa'] = value
        if key == 'Kecamatan':
            parsed_data['Kecamatan'] = removedigit(value)
        if key == 'Agama':
            removedigit(value)
            if 'ISLAM' in value:
                parsed_data['Agama'] = 'ISLAM'
            elif "KRISTEN" in value:
                parsed_data['Agama'] = 'KRISTEN'
            elif 'KATHOLIK' in value:
                parsed_data['Agama'] = 'KATHOLIK'
            elif 'HINDU' in value:
                parsed_data['Agama'] = 'HINDU'
            elif 'BUDDHA' in value:
                parsed_data['Agama'] = 'BUDDHA'
            elif 'KONGHUCU' in value:
                parsed_data['Agama'] = 'KONGHUCU'
            else:
                parsed_data['Agama'] = '-'
        
        if key == 'Status' or key == 'Perkawinan':
            if 'BELUM' in value and 'KAWIN' in value :
                parsed_data['Status_Perkawinan'] = "BELUM KAWIN"  
            elif 'CERAI' in value and 'HIDUP' in value:
                parsed_data['Status_Perkawinan'] = "CERAI HIDUP"
            elif 'CERAI' in value and 'MATI' in value:
                parsed_data['Status_Perkawinan'] = "CERAI MATI"
            elif 'KAWIN' in value:
                parsed_data['Status_Perkawinan'] = "KAWIN"
            else:
                parsed_data['Status_Perkawinan'] = "-"
                
        if key == 'Pekerjaan':
            parsed_data['Pekerjaan'] = removedigit(value)
            
        if key == 'Kewarganegaraan':
            removedigit(value)
            if "WNI" in value:
                parsed_data['Kewarganegaraan'] = "WNI"
            elif "WNA" in value:
                parsed_data['Kewarganegaraan'] = "WNA"
            else:
                parsed_data['Kewarganegaraan'] = "-"
                
        if key == 'Berlaku' or key == 'Hingga':
            parsed_data['Berlaku_Hingga'] = value
            
    return parsed_data        
    
def removedigit(data):
    return re.sub(r'\d', '', data)

def main(image):
    raw_df = pd.read_csv(LINE_REC_PATH, header=None)
    cities_df = pd.read_csv(CITIES_REC_PATH, header=None)
    religion_df = pd.read_csv(RELIGION_REC_PATH, header=None)
    marriage_df = pd.read_csv(MARRIAGE_REC_PATH, header=None)
    jenis_kelamin_df = pd.read_csv(JENIS_KELAMIN_REC_PATH, header=None)
    province_df = pd.read_csv(PROVINCE_REC_PATH, header=None)
    distric_df = pd.read_csv(DISTRIC_REC_PATH, header=None)
    pekerjaan_df = pd.read_csv(PEKERJAAN_REC_PATH, header=None)
    result_raw, id_number = ocr_raw(image)
    result_list = strip_op(result_raw)
    data = []
    # print("NIK: " + str(id_number))

    loc2index = dict()
    for i, tmp_line in enumerate(result_list):
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word_, tmp_word.strip(':')) for tmp_word_ in raw_df[0].values]

            tmp_sim_np = np.asarray(tmp_sim_list)
            arg_max = np.argmax(tmp_sim_np)

            if tmp_sim_np[arg_max] >= 0.6:
                loc2index[(i, j)] = arg_max

    last_result_list = []
    useful_info = False

    for i, tmp_line in enumerate(result_list):
        tmp_list = []
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_word = tmp_word.strip(':')

            if(i, j) in loc2index:
                useful_info = True
                if loc2index[(i, j)] == NEXT_LINE:
                    last_result_list.append(tmp_list)
                    tmp_list = []
                tmp_list.append(raw_df[0].values[loc2index[(i, j)]])
                if loc2index[(i, j)] in NEED_COLON:
                    tmp_list.append(':')
            elif tmp_word == ':' or tmp_word =='':
                continue
            else:
                tmp_list.append(tmp_word)

        if useful_info:
            if len(last_result_list) > 2 and ':' not in tmp_list:
                last_result_list[-1].extend(tmp_list)
            else:
                last_result_list.append(tmp_list)
    # print(last_result_list)
    for tmp_data in last_result_list:
        if '—' in tmp_data:
            tmp_data.remove('—')

        if 'PROVINSI' in tmp_data or 'KABUPATEN' in tmp_data or 'KOTA' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in province_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 1] = province_df[0].values[arg_max]

        if 'KABUPATEN' in tmp_data or 'KOTA' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in cities_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 1] = cities_df[0].values[arg_max]
                    
        if 'Nama' in tmp_data:
            nama = ' '.join(tmp_data[2:])
            nama = re.sub('[^A-Z. ]', '', nama)

            if len(nama.split()) == 1:
                nama = re.sub('[^A-Z.]', '', nama)

        if 'NIK' in tmp_data:
            if len(id_number) != 16:
                # id_number = tmp_data[2]

                if "D" in id_number:
                    id_number = id_number.replace("D", "0")
                if "?" in id_number:
                    id_number = id_number.replace("?", "7")
                if "L" in id_number:
                    id_number = id_number.replace("L", "1")
                if "I" in id_number:
                    id_number = id_number.replace("I", "1")
                if "R" in id_number: 
                    id_number = id_number.replace("R", "2")
                if "O" in id_number:
                    id_number = id_number.replace("O", "0")   
                if "o" in id_number:
                    id_number = id_number.replace("o", "0")
                if "S" in id_number:
                    id_number = id_number.replace("S", "5")
                if "G" in id_number:
                    id_number = id_number.replace("G", "6")

                while len(tmp_data) > 2:
                    tmp_data.pop()
                tmp_data.append(id_number)
            else:
                while len(tmp_data) > 3:
                    tmp_data.pop()
                if len(tmp_data) < 3:
                    tmp_data.append(id_number)
                tmp_data[2] = id_number

        if 'Agama' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in religion_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
 
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 1] = religion_df[0].values[arg_max]
                    
                    
        if 'Status' in tmp_data or 'Perkawinan' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[2:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in marriage_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
 
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 2] = marriage_df[0].values[arg_max]
                    
                # tmp_data = removedigit(tmp_data)
                    
            
        if 'Alamat' in tmp_data:
            for tmp_index in range(len(tmp_data)):
                if "!" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
                if "1" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
                if "i" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
                if "RI" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("RI", 'RT')
                if "Rw" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("Rw", 'RW')
                if "rw" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("rw", 'RW')
                if "rt" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("rt", 'RT')

        if 'RT/RW' in tmp_data or 'RT' in tmp_data or 'RW' in tmp_data:
            tmp_data = [item for item in tmp_data if item != ""]
            for index, elemen in enumerate(tmp_data):
                if '“' in elemen:
                    tmp_data[index] = tmp_data[index].replace('“', '')
                if '"' in elemen:
                    tmp_data[index] = tmp_data[index].replace('"', '')
                if 'f' in elemen:
                    tmp_data[index] = tmp_data[index].replace('f', '/')
                if elemen.startswith('/') or elemen.endswith('/') and elemen[1:].isdigit():
                    tmp_data[index] = elemen.replace('/', '')
                if re.match(r'^(\d{3})(\d{3})$', elemen):
                    tmp_data[index] = re.sub(r'^(\d{3})(\d{3})$', r'\1/\2', elemen)
    
                tmp_data = [re.sub(r'^\d{4,}', lambda x: x.group()[1:], item) for item in tmp_data]
                if '/' not in tmp_data and len(tmp_data) <= 3:
                    clean_ = [item for item in tmp_data if re.match(r'\d{3}', item)]
                    clean_ = [i.split('/') for i in clean_]
                    clean_[0] = '/'.join(clean_[0])
                    index = tmp_data.index(':')
                    tmp_data = tmp_data[:index+ 1]
                    tmp_data = tmp_data + clean_

                if '/' not in tmp_data and len(tmp_data) > 3:
                    tmp_data.insert(3, '/')
                    tmp_data = [x for x in tmp_data if x != '']

        if 'Jenis' in tmp_data or 'Kelamin' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[2:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in jenis_kelamin_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)

                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 2] = jenis_kelamin_df[0].values[arg_max]
        
        if 'Gol.' in tmp_data or ' Darah' in tmp_data or 'Darah' in tmp_data:
            if len(tmp_data) > 3:
                if tmp_data[3] == '0':
                        tmp_data[3] = tmp_data[3].replace('0', 'O')
                if tmp_data[3] == '8':
                        tmp_data[3] = tmp_data[3].replace('8', 'B')
            else:
                tmp_data.insert(3, '-')  

        if 'Tempat' in tmp_data or 'Tgl' in tmp_data or 'Lahir' in tmp_data:
            join_tmp = ' '.join(tmp_data)

            match_tgl1 = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", join_tmp)
            match_tgl2 = re.search("([0-9]{2}\ [0-9]{2}\ [0-9]{4})", join_tmp)
            match_tgl3 = re.search("([0-9]{2}\-[0-9]{2}\ [0-9]{4})", join_tmp)
            match_tgl4 = re.search("([0-9]{2}\ [0-9]{2}\-[0-9]{4})", join_tmp)

            if match_tgl1:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl1.group(), '%d-%m-%Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl2:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl2.group(), '%d %m %Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl3:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl3.group(), '%d-%m %Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl4:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl4.group(), '%d %m-%Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            else:
                tgl_lahir = ""

            for tmp_index, tmp_word in enumerate(tmp_data[2:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in cities_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
  
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 2] = cities_df[0].values[arg_max]
                    tempat_lahir = tmp_data[tmp_index + 2]
                    
        if 'Kecamatan' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in distric_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
                
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 1] = distric_df[0].values[arg_max] 
          
        
        if 'Pekerjaan' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in pekerjaan_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
                
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 1] = pekerjaan_df[0].values[arg_max]
                    
                      
        data.append(tmp_data)
    clean = parsing(data) 
    return clean

if __name__ == '__main__':
    main(sys.argv[1])

