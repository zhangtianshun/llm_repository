import os

from dotenv import load_dotenv

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
print(DEEPSEEK_API_KEY)

MILVUS_URI = 'http://192.168.224.1:19530'

ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')
ZHIPU_API_URI = os.getenv('ZHIPU_API_URI')

os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

