for c_row in $(seq 1 3); do

    c_fname=$(awk 'NR=='$c_row' {print $1}' my_igm_params.dat)
    # cparam=(awk 'NR==$crow, {print, $2}' my_igm_params.txt)
    c_value=$(awk 'NR=='$c_row' {print $3}' my_igm_params.dat)

    for a_row in $(seq 4 6); do

        a_fname=$(awk 'NR=='$a_row' {print $1}' my_igm_params.dat)
        # aparam=(awk 'NR==$crow, {print, $2}' my_igm_params.txt)
        a_value=$(awk 'NR=='$a_row' {print $3}' my_igm_params.dat)

        replaceparam="output/synthetic_forward"$c_fname$a_fname".nc"
        echo $replaceparam

        # sed -i '/iflo.save-file/c $replaceparam' /temp/foo
        igm_run --iflo_init_slidingco $c_value --iflo_init_arrhenius $a_value --wncd_output_file $replaceparam | tee -a output/synthetic_forward_linSurf_igm_output.txt
        wait

    done
done