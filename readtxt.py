#github에 올려야 함
with open("BostonData.txt", "r") as f:
     for line in f:
         print(line.strip())

#홀수번째 줄만 읽고, 그 줄에서 6번째 데이터만 가져오고 싶음(input_array)
with open("BostonData.txt", "r") as f: #처리할 txt 파일을 읽기 모드로 읽음
    lines = f.readlines() #각 라인들을 "lines"라는 리스트에 넣음. 각 줄이 리스트 멤버가 된 것 
    for line in lines: # "lines"의 멤버들을 "line"이라는 변수에 넣어서 반복 계산을 하겠다
        print(line.strip())

    