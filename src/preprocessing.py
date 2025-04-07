import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(csv_path='data/raw/weather.csv'):

    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)

    # xử lý nhãn, dữ liệu thiếu và chuyển đổi kiểu dữ liệu.
    # Nếu dữ liệu thiếu => loại bỏ. Không thay thế bằng giá trị trùng bình, giá trị xuất hiện nhiều nhất,.....
    df.columns = ['province', 'max_temp', 'min_temp', 'wind_speed', 'wind_dir',
                  'rainfall', 'humidity', 'cloud', 'pressure', 'date']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(inplace=True)

   # có thể bỏ  việc lọc trội
    # #sử dụng IQR lloại trội ngoài khoảng [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    Q1 = df['rainfall'].quantile(0.25)
    Q3 = df['rainfall'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['rainfall'] >= (Q1 - 1.5 * IQR)) & (df['rainfall'] <= (Q3 + 1.5 * IQR))]

    # Chia cột rainfall thành các mức độ mưa
    # Linh hoạt theo việc chia, vì dữ liệu đầu vào có vấn đề
    # Chia thành 2 nhóm: có mưa và không mưa, >= 2mm là có mưa
    def categorize_rainfall(rainfall):
        if rainfall >= 2:
            return 'Mưa '
        else:
            return 'Không mưa'
        
     # 'Mưa' 1, 'Không mưa' 0
    df['rain_category'] = df['rainfall'].apply(categorize_rainfall)
    le = LabelEncoder()
    y = le.fit_transform(df['rain_category']) 





    df = pd.get_dummies(df, columns=['wind_dir'], drop_first=True)
    #Hướng gió được chuyển đổi thành các biến nhị phân (0 hoặc 1) cho mỗi hướng gió.
    # 
    #    'wind_dir_ENE', 'wind_dir_ESE', 'wind_dir_N', 'wind_dir_NE',
    #    'wind_dir_NNE', 'wind_dir_NNW', 'wind_dir_NW', 'wind_dir_S',
    #    'wind_dir_SE', 'wind_dir_SSE', 'wind_dir_SSW', 'wind_dir_SW',
    #    'wind_dir_W', 'wind_dir_WNW', 'wind_dir_WSW',

    # Chọn đặc trưng và chuẩn hóa. bỏ target và các cột không cần thiết
    #Theo rain  (target variable), chia thành 2 nhóm: có mưa và không  mưa.
    feature_cols = [col for col in df.columns if col not in ['rainfall', 'rain_category', 'date', 'province']]
    X = df[feature_cols]
    y = df['rain_category']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)


    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# test
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    # Lưu X_train, X_test, y_train, y_test vào thư mục processed/
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    X_train_df.to_csv('data/processed/X_train.csv', index=False)
    X_test_df.to_csv('data/processed/X_test.csv', index=False)
    y_train_df.to_csv('data/processed/y_train.csv', index=False)
    y_test_df.to_csv('data/processed/y_test.csv', index=False)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train value counts:\n", y_train.value_counts())
    print("y_test value counts:\n", y_test.value_counts())

