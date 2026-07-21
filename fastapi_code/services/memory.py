import redis.asyncio as redis
import json

redis_client = redis.Redis(
    host="redis",
    port=6379,
    decode_responses=True
)

MAX_HISTORY = 50


async def get_history(key):
    #redis的list类型
    #取最近的数据-1  (key, start, end)   
    #没有rrange   按索引   
    history = await redis_client.lrange(key, -MAX_HISTORY, -1)
    return [json.loads(m) for m in history]


async def save_message(key, message):
    # pipe = redis_client.pipeline()

    # pipe.rpush(key, json.dumps(message, ensure_ascii=False))
    # #最后一个参数  -1：保留最近的数据   0：保留最老的数据
    # pipe.ltrim(key, -MAX_HISTORY, -1)

    # await pipe.execute() 最新的在最右边（rpush）  
    await redis_client.rpush(key, json.dumps(message, ensure_ascii=False))
    #ltrim(name, start, end)  选列表的范围  start end 表示的是索引
    await redis_client.ltrim(key, -MAX_HISTORY, -1)


