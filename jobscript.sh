#BSUB -J pypsa-eur
#BSUB -o log_%J.out
#BSUB -e log_%J.err
#BSUB -q p_macro
#BSUB -n 32
#BSUB -P 0588
#BSUB -W 1440
#BSUB -x
#BSUB -R "span[ptile=32]"
#BSUB -R "rusage[mem=500G]"

export TMPDIR=/work/cmcc/fp01525/tmp
mkdir -p "$TMPDIR"
export PATH="/users_home/cmcc/fp01525/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=/work/cmcc/fp01525/.cache/pixi
export XDG_CACHE_HOME=/work/cmcc/fp01525/.cache

pixi run snakemake --cores 32\
    resources/pypsa-macro-italy/networks/base_s_adm___2025_brownfield.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2030.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2035.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2040.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2045.nc \
    resources/pypsa-macro-italy/networks/base_s_adm___2050.nc \
    --configfile config/config.macro.italy.yaml