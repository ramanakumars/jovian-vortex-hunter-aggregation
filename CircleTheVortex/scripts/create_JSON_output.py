from aggregation import Aggregator


aggregator = Aggregator('reductions/shape_reducer_hdbscan.csv')
aggregator.save_JSON('reductions/data.json')
