import json
import os.path
import sys
import tqdm
import multiprocessing
import signal
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from aggregation import Aggregator
    from aggregation.workflow import NpEncoder
    from aggregation.vortex import MultiSubjectVortex
    from aggregation.vortex_cluster import (cluster_vortices,
                                            average_vortex_cluster)
except Exception:
    pass


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# load the classification data
aggregator = Aggregator.from_JSON('reductions/data.json')
aggregator.load_subject_data('../subjects_data.csv')

# get the unique list of vortices
ellipses = aggregator.get_ellipses(sigma_cut=0.8, prob_cut=0.)

print()

avg_ellipses = []

pbar = tqdm.tqdm(total=(36-13)*4, desc='Finding vortices', position=0)

for PJ in range(13, 36):
    for key in ['white', 'red', 'brown', 'dark']:
        # find the ellipses in this perijove which correspond
        # to this color
        ellipses_i = list(filter(lambda e: e.perijove == PJ,
                                 ellipses[key]))

        # get the clusters and lone vortices
        lone, groups = cluster_vortices(ellipses_i)

        pbar.set_postfix_str(f"{key}: {len(ellipses_i)} => "
                             f"{len(groups)} gr, {len(lone)} ln")

        with multiprocessing.Pool(processes=8,
                                  initializer=initializer) as pool:
            inp_args = []
            # get the list of extracts in the different groups
            for ell_group in groups:
                ells_ext = []
                _ = [ells_ext.extend(ell.extracts) for ell in ell_group]

                inp_args.append(ells_ext)

            try:
                # apply the extracts on the cluster averaging function
                r = pool.map_async(average_vortex_cluster, inp_args)

                pool.close()

                # print out a progress bar
                tasks = pool._cache[r._job]
                ninpt = len(inp_args)

                with tqdm.tqdm(total=ninpt,
                               position=1,
                               leave=False) as pbar_inner:
                    while tasks._number_left > 0:
                        pbar_inner.n = max([ninpt - tasks._number_left *
                                            tasks._chunksize, 0])
                        pbar_inner.refresh()

                        time.sleep(0.5)
            except Exception as e:
                print(e)
                pool.terminate()
                pool.join()

            pool.join()

            # get the results from the multiprocessing queue
            results = r.get()

        # add the results back to the original list of vortices
        for res in results:
            avg_ellipses.append(res)

        for ell in lone:
            # lone vortices will be converted to a multisubjectvortex
            # but with a list of one subject_id
            avg_ell = MultiSubjectVortex(ell.ellipse_params,
                                         ell.sigma,
                                         ell.lon0, ell.lat0,
                                         ell.x0, ell.y0)

            avg_ell.perijove = ell.perijove
            avg_ell.color = ell.color
            avg_ell.extracts = ell.extracts
            avg_ell.subject_ids = [ell.subject_id]

            avg_ellipses.append(avg_ell)

        pbar.update(1)

# save out the data
with open('reductions/vortices.json', 'w') as outfile:
    json.dump([e.as_dict() for e in avg_ellipses], outfile,
              cls=NpEncoder)
