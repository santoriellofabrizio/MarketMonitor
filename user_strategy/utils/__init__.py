from joblib import Memory
from pandas._libs.tslibs.offsets import CustomBusinessDay

memoryPriceProvider = Memory(location='.cache/cache_db_connection', verbose=0)
HOLIDAY = ['2024-12-24', '2024-12-25', '2024-12-26', '2024-12-31', '2025-01-01', '2025-01-06', '2025-08-15','2025-10-29']


CustomBDay = CustomBusinessDay(holidays=HOLIDAY, weekmask="Mon Tue Wed Thu Fri")