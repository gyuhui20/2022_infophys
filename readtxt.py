#github에 올려야 함
import numpy as np

room_number_list, price_list = [], []

with open("BostonData1.txt", "r") as f:
     for lines in f:
        a = lines[0:5]
        b = lines[5:10]
        room_number_list.append(float(a.replace(" ", "")))
        price_list.append(float(b.replace(" ", "")))
        #print(type(a))
        #b=np.array([a])
        #print(b.T)
#print(room_number_list)

room_number_array = np.array(room_number_list)
price_array = np.array(price_list)

print(room_number_array.shape)
#print(b.shape)
#with open("BostonData1.txt", "r") as f: #처리할 txt 파일을 읽기 모드로 읽음
        #lines = f.readlines() #각 라인들을 "lines"라는 리스트에 넣음. 각 줄이 리스트 멤버가 된 것 
        #for line in lines: # "lines"의 멤버들을 "line"이라는 변수에 넣어서 반복 계산을 하겠다
            #input_array=line.strip()[0:5]
            #target_array=line.strip()[5:10]
#홀수번째 줄만 읽고, 그 줄에서 6번째 데이터만 가져오고 싶음(input_array)
# with open("BostonData1.txt", "r") as f: 
#     line = f.readline
#     print(line)
