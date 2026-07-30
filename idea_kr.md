# KOR (수동 작성)
## 로직
1. 음향파일에서 이 파일이 너무 짧은지 확인 함.
2. 음향파일에서 음성이 있는지 확인함.
   1. 음향에서 초반,후반 15%는 대사가 아닌 슬레이트,감독,다른것으로 채워져 있음으로 룸톤 구분에 사용하지 않음.
   2. 음성이 없는 파일은 자동으로 분류함. 룸톤 구분
3. 음향파일에서 음성이 있는 구간을 찾아서 음성 구간만 잘라냄.
   1. 잘라낸 구간 +- 1초 정도를 붙여 stt모델에 사용함 
4. 음성 구간만 잘라낸 파일을 STT 모델에 넣어서 텍스트로 변환함.
5. 텍스트를 사용하여 자동 분류
6. 미분류된것들은 수동탭에 분류됨
7. 자동분류된것들은 최종검수탭에 분류됨
## 필요한 탭
1. 자동분류탭
    1. 자동분류 과정 검수 및 오리지널 파일 선택
2. 수동분류탭
   1. 자동분류되지 않은 파일들을 수동으로 분류함.  
   2. 자동분류된 파일들은 수동분류탭에 나타나지 않음.
3. 최종검수탭
   1. 자동분류된 파일들을 최종검수함.
4. 음성 강화탭
   1. 음성 강화 모델이나 분리 모델들을 활용하여 사용함.
5. 설정탭
   1. 기본 설정등을 확인함
6. 검수탭
    1. 오리지널 파일들과 대조해서 해쉬로 검수해서 다 옮겨진지 파일 사이즈를 보고 검수후 .csv 파일이나 .pdf파일로 저장함
## 구현 순서
1. UI 구현
2. 로직 구현
    1. 파일 불러오기 구현
    2. VAD 모델 사용방법 구현
    3. STT 모델 사용방법 구현
    4. 로그 확인
3. 설정 구현
   1. 필요설정 정리
   2. 설정 탭 분할
   3. 사용성에 맞게 배치
   4. 사용가능한지 확인
   5. 설정 적용
   6. 설정 저장
   7. 설정 불러오기
   8. 설정 초기화
4. 최종 검수 구현
   1. 1개씩 들어보고 확인 재검수(자동),
   2. 수동 재검수(수동 입력후 폴더 이동)을 구현
5. 음성 강화 구현
   1. 음성 강화 모델이나 분리 모델들을 활용하여 사용함.
   2. mix비율 게인 조절등을 추가
6. 검수 구현
   1. 오리지널 파일에 해쉬 값 비교해서 검수
   2. CSV로 기본저장
   3. pdf 익스포트 기능 만들기
# ENG (use google translate) long update time
## Logic
1. Check if the audio file is too short.

2. Check if there is voice in the audio file.
   1. The beginning and end 15% of the audio are filled with slates, the director, and other elements rather than dialogue, so they are not used for room tone classification.
   2. Files without voice are automatically classified. Room tone classification.
3. Find sections with voice in the audio file and cut out only the voice segments.
   1. Append approximately ± 1 second to the cut segment and use it in the STT model.
4. Input the file containing only the voice segments into the STT model to convert it into text.
5.   Automatic classification using the text.
6. Unclassified items are classified in the Manual tab.
7. Automatically classified items are classified in the Final Inspection tab.
## Required Tabs

1. Automatic Classification Tab

   1. Inspect the automatic classification process and select the original file.

2. Manual Classification Tab

   1. Manually classify files that were not automatically classified. 2. Automatically classified files do not appear in the Manual Classification tab.

3. Final Inspection Tab

   1. Perform a final inspection of automatically classified files.

4. Voice Enhancement Tab

   1. Utilize voice enhancement models or separation models.

5. Settings Tab

    1. Check basic settings, etc.

6. Verification Tab

   1. Verify by comparing with original files and checking hashes to confirm all files have been transferred. After verification, save as a .csv or .pdf file.
## Implementation Order
1. UI Implementation
2. Logic Implementation
    1. File Loading Implementation
    2. VAD Model Usage Implementation
    3. STT Model Usage Implementation
   1. Log Check
3. Settings Implementation
    1. Organize necessary settings
    2. Split Settings Tab
    3. Arrange for usability
    4. Check availability
    5. Apply settings
    6. Save settings
    7.  Load settings
    8.  Reset settings
4.  Final Verification Implementation
    1.  Re-verify by listening to and confirming one by one (Automatic),
    2.  Implement manual re-verification (move folder after manual input)
5.  Voice Enhancement Implementation
    1.  Utilize voice enhancement models or separation models. 2. Add mix ratio, gain control, etc.
    2. Implement inspection
6. Inspection by comparing hash values ​​with the original file
    1. Save as CSV by default
    2. Create PDF export function
