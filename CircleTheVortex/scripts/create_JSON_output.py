import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from aggregation import Aggregator
except Exception:
    pass


aggregator = Aggregator('../subjects_data.csv', 'reductions/shape_reducer_dbscan_reductions.csv', autoload=True)
aggregator.save_JSON('reductions/data.json')
