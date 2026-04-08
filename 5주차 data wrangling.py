import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

#Read the data
df = pd.read_csv('MDA_05_dirty_data.csv')
print(df.head())   #상위 5개 데이터 출력

#describe data(기본적인 statastics)
print(df.describe())

#info()   data의 정보 출력(Type은 뭐고, nonnull이 몇개 있는지)
print(df.info())

#WESF에는 Null이 아닌 11row가 있기에 확인 하는 코드
print(df[df.WESF.isna()==False])

#적어도 하나의 Null값을 가지는 row 모두 출력
contain_nulls = df[df.SNOW.isna() | df.SNWD.isna() | df.TOBS.isna()
                   | df.WESF.isna() | df.inclement_weather.isna()]

print(contain_nulls.shape[0])   #몇개의 row가 있는지 확인
print(contain_nulls.head(10))  #상위 10개 출력

#find inf/-inf in each column
def get_inf_count(df):
    return {col: df[df[col].isin([np.inf, -np.inf])].shape[0] for col in df.columns}   #df의 모든 column를 for문 돌려 inf인지 -inf인지 row를 count하여 return

print(get_inf_count(df))

#Snow와 SNWD가 연관되어 있으니 SNWD가 inf인것과 -inf인것을 뽑고 snow를 describe해보자
snow = pd.DataFrame({
    'np.inf Snow Depth': df[df.SNWD == np.inf].SNOW.describe(),
    '-np.inf Snow Depth': df[df.SNWD == -np.inf].SNOW.describe()
})
print(snow)   #여기서 눈의 깊이가 -inf일때는 눈이 안왔다는 것을 알수있음

#Understanding 'date' and 'station'
print(df.describe(include='object'))    #문자열 타입 데이터 요약(count, unique, top, freq)
#365일 데이터인데 중복된 값이 있음

#중복확인
print(df[df.duplicated(keep='first')].head(20))   #중복된 값 상위 20개만 출력, 만약 first이면 중복된 값은 첫번쨰 1개만 제외 나머지는 출력


#data wrangling
#1 make the date a datetime (dont need hours)
df.date = pd.to_datetime(df.date)

#2 save the WESF information
#station이 ?인것 중에 dates가 겹치는것은 1개만 빼고 제거하고 WESF Column만 추출
station_qm_wesf = df[df.station == "?"].drop_duplicates('date').set_index('date').WESF  

#3 sort "?" to the bottom
#station column을 내림차순으로 해서 ?를 아래로 보내고, 원본 데이터를 바꿈
df.sort_values('station', ascending=False, inplace=True)    

#4 drop duplicates based on the date column and keep first one
df_deduped = df.drop_duplicates("date")
print(df_deduped)

#5 remove the station column because we dont need it now
df_deduped = df_deduped.drop(columns='station').set_index('date').sort_index()

#6 insert the saved WESF value
df_deduped = df_deduped.assign(
    WESF=lambda x:x.WESF.combine_first(station_qm_wesf)
)
print(df_deduped)
print(df_deduped.shape)

#####
#Handling Nulls - Remove
print(df_deduped.dropna())  #drop Nulls

df_deduped = df_deduped.dropna(
    how='all', subset=['inclement_weather', 'SNOW', "SNWD"]   #위 3개 모두 NULL갑이면 그 row는 버려라
)
print(df_deduped)

#Remove only if there are nulls in all columns
df_deduped = df_deduped.dropna(how='all')
print(df_deduped.shape)


########
#filling

#filling with constant value
print(df_deduped['WESF'].fillna(0, inplace=True))   #Null값들을 모두 0으로 바꿔라

#inf 와 -inf를 Null로 변경
df_deduped = df_deduped.assign(
    TMAX=lambda x:x.TMAX.replace(5505, np.nan),
    TMIN=lambda x:x.TMIN.replace(-40, np.nan)
)
print(df_deduped)


#앞이나 뒤에 있는 값 똑같이 넣기
df_deduped = df_deduped.assign(
    TMAX=lambda x:x.TMAX.ffill(),      #ffill은 앞에 있는 값 넣기
    TMIN=lambda x:x.TMIN.bfill()      #bfill은 뒤에 있는 값 넣기
)
print(df_deduped)


#clip을 사용하여 null 값을 min 또는 max 값으로 넣기
df_deduped = df_deduped.assign(
    SNWD=lambda x: x.SNWD.clip(0, x.SNOW)
)

#fillna()를 사용하여 계산된 값을 NULL값에 넣음
df_deduped = df_deduped.assign(
    TMAX=lambda x:x.TMAX.fillna(x.TMAX.median()),
    TMIN=lambda x:x.TMIN.fillna(x.TMIN.median()),
    #average
    TOBS=lambda x:x.TOBS.fillna((x.TMAX + x.TMIN)/2)  #TOBS의 null값에는 그 row의 TMAX와 TMIN의 평균으로 넣음
)

#Interpolate
#2018-01-01에서 2018-12-31까지 비어있는 날짜가 있으면 생성하고 생성된 날짜의 데이터들은 interpolate해서 채워준다  (기본값은 linear)
df_deduped=df_deduped.reindex(
    pd.date_range('2018-01-01', '2018-12-31', freq="D")).apply(lambda x:x.interpolate())