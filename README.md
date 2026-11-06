# making_the_graph

Notebooks and scripts live at the top level and are run from this folder
(all paths inside them are relative to here).

```
data/
  colorado/        MERIT rasters (.tif), WQ measurements, station tables
  chattahoochee/   MERIT rasters, basin/HUC8 geojson, hourly WQ + station metadata
  usgs/            USGS download plans/audits, combined hourly CSVs,
                   and per-station folders (raw, hourly, cleaned, final_wq_*)
  benchmark_data/  model-ready inputs for the benchmark (static/fourier dfs, laplacians)
graphs/
  colorado/        NetworkX graphs (.pkl / .graphml): full, largest component, compressed variants
  chattahoochee/   Chattahoochee graphs (full, compressed 10px/100px, structural)
results/           model result CSVs (LOO, Laplacian vs linear, ...) + benchmark_results/
figures/           plots (.png) and the interactive map HTML
docs/              slides (beamer PDF)
share/             bundles prepared for sharing with the lab
```
