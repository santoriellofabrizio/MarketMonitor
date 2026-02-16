"""
This directory contains all the thread used to produce data:
    - BloombergStreamingThread: thread used to populate a book shared with the strategy.
    - KafkaStreamingThread: thread used to receive book/price data from Kafka (e.g. BookBest topics).
    - KafkaTradeStreamingThread: thread used to receive trade data from Kafka (PublicDeal, Trade topics)
      and route them as (TradeType, pd.DataFrame) tuples to a shared Queue.
    - trade: thread used to insert trades in a queue shared with the strategy.
    - excel: thread used to read from an open excel file.
"""