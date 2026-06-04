#BSUB -J pypsa-eur
#BSUB -o log_%J.out
#BSUB -e log_%J.err
#BSUB -q p_macro
#BSUB -n 16
#BSUB -P 0588
#BSUB -W 1440
#BSUB -x
#BSUB -R "span[ptile=16]"
#BSUB -R "rusage[mem=500G]"

export XDG_CACHE_HOME=/work/cmcc/fp01525/.cache
export HDF5_USE_FILE_LOCKING=FALSE
pixi shell
snakemake -call \
    resources/pypsa-macro-italy/networks/base_s_adm___2025_brownfield.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2030.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2035.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2040.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2045.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2050.nc \
    --configfile config/config.macro.italy.yaml