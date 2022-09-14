import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from aggregation import Aggregator
except Exception:
    pass


aggregator = Aggregator('reductions/shape_reducer_hdbscan_reductions.csv')
aggregator.save_JSON('reductions/data.json')
