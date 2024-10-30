#!/bin/sh 

field=$1   # variable
e=$2       # ensemble
x=$3
force=y

nckspre="ncks -O -L 0 --chunk_policy g3d --cnk_dmn plev,1 --cnk_dmn rlon,50 --cnk_dmn rlat,50 --cnk_dmn time,1 "
nckspost="ncks -O --chunk_policy g3d --cnk_dmn time,31 "

if [ $x == historical ]; then
    sy=1955
    ly=1955
elif  [ $x == rcp85 ]; then
    sy=2006
    ly=2098
fi

y=$sy
while [ $y -le $ly ]; do
    ddir=/home/portal/work_big/CRCM5-LE/${field}/$e/$y
    echo $ddir
    
    for ff in `ls ${ddir}/${field}*${y}*.nc`; do
	echo $ff
	nustart=`echo $ff | rev | cut -d _ -f 2 | rev` # e.g. 1h/3h
	ffuz=${ff/.nc/_unzip.nc}
	ffuzmean=${ffuz/$nustart/daymean}
	ffmean=${ff/$nustart/daymean}

	if [[ ! -f $ffmean  || $force == y ]]; then
	    start=`date +%s`
	    $nckspre $ff $ffuz
	    cdo -daymean $ffuz $ffuzmean  # BAD CONVERSION - ONLY FOR CONSISTENCY WITH F MARRA
            mid=`date +%s`
	    $nckspost $ffuzmean $ffmean
            end=`date +%s`
	    echo $((mid-start))
	    echo $((end-start))
	#    cdo -v mulc,86400 -daymean ${ddir}/tmp/$ffuz ${ddir}/tmp/$ffmean  # convert to accumulated mm/day
	fi

	rm -f $ffuz $ffuzmean
    done
    
    y=$(($y+1))
done

# ## merge data
# if [ $field == pr ]; then
#     nustart=1h
# elif [ $field == prc ]; then
#     nustart=3h   
# fi
#     
# mkdir -p ${field}/daily
# if [ $nustart == 1h ] ;then
#     nustartdir=hourly
# elif  [ $nustart == 3h ] ;then
#     nustartdir=threehourly
# fi
# mkdir -p ${field}/${nustartdir}
# 
# timep=${sy}01-${ly}12
# 
# # merge maps
# nu=('daysum' "${nustart}-daymax")
# for n in ${nu[@]}; do
#     fmerge=(`ls ${field}/${e}/*/tmp/*${x}*${n}*`)
#     fout=${field}_${n}_${x}_${timep}_${e}.nc
#     if [[ $force == y && -f ${field}/daily/${fout} ]]; then
# 	rm -f ${field}/daily/${fout}
#     elif [[ $force == n && -f ${field}/daily/${fout} ]]; then
# 	continue
#     fi
# 
#     if [ ${#fmerge[@]} -le 1000 ]; then
# 	cdo -z zip mergetime ${fmerge[@]} ${field}/daily/${fout}
#     elif  [ ${#fmerge[@]} -le 2000 ]; then
# 	cdo mergetime ${fmerge[@]:0:1000} ${field}/daily/${fout}_1
# 	cdo mergetime ${fmerge[@]:1000:1000} ${field}/daily/${fout}_2
# 	cdo -z zip mergetime ${field}/daily/${fout}_1 ${field}/daily/${fout}_2 ${field}/daily/${fout}
# 	rm ${field}/daily/${fout}_1 ${field}/daily/${fout}_2
#     fi
#     if [ $n == 'daysum' ]; then
# 	ncatted -O -a units,${field},m,c,'mm/day' ${field}/daily/${fout} 
#     fi
#     rm $fmerge
# done


exit 0
