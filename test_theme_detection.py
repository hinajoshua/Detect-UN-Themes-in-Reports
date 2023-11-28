import theme_detection
import pandas as pd
df=pd.read_csv(r'rw_themes_multilabeldf.csv')
text = df['text'].iloc[100]

themes_detected = theme_detection.detect_theme(text,'Model_RW_ThemeDetect.pkl', 
                             'Vectorizer_RW_ThemeDetect.pkl', 
                             theme_detection.themes_list() )

print(themes_detected)

themes = (theme_detection.themes_list())
print(df[df[themes] == 1].iloc[100].dropna())