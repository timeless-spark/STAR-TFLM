function wait_enter {
    # echo "Press the enter to go to next test"
    # while true; do
    # read -s -n 1 key
    # if [[ $key = "" ]]; then
    #     break
    # fi
    # done

    echo "Wait for 15 mins"
    sleep 15m
}

# In totale sono 156 test

############################# standard int8 global #############################

make soft-tflm-clean

# # standard (int8)
#     # global
#     echo "standard imclass int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

# standard (int8)
    # global
    echo "standard mobilenet int8 global" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=global soft
    make fpga-program
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
    wait_enter

# # standard (int8)
#     # global
#     echo "standard keyword int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

# # standard (int8)
#     # global
#     echo "standard anomaly int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

############################# standard int8 kernel #############################

make soft-tflm-clean

# # standard (int8)
#     # kernel
#     echo "standard imclass int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# standard (int8)
    # kernel
    echo "standard mobilenet int8 kernel" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
    make fpga-program
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
    wait_enter

# # standard (int8)
#     # kernel
#     echo "standard keyword int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # standard (int8)
#     # kernel
#     echo "standard anomaly int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

############################# standard int8 only_cnn #############################

make soft-tflm-clean

# # standard (int8)
#     # only_cnn
#     echo "standard imclass int8 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# standard (int8)
    # only_cnn
    echo "standard mobilenet int8 only_cnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
    make fpga-program
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
    wait_enter

# # standard (int8)
#     # only_cnn
#     echo "standard keyword int8 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

############################# standard int8 only_dwcnn #############################

make soft-tflm-clean

# standard (int8)
    # only_dwcnn
    echo "standard mobilenet int8 only_dwcnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn soft
    make fpga-program
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn fpga-run
    wait_enter

# # standard (int8)
#     # only_dwcnn
#     echo "standard keyword int8 only_dwcnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn fpga-run
#     wait_enter

############################# standard int8 only_fc #############################

make soft-tflm-clean

# # standard (int8)
#     # only_fc
#     echo "standard imclass int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# standard (int8)
    # only_fc
    echo "standard mobilenet int8 only_fc" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
    make fpga-program
    make TFLM_TYPE=standard TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
    wait_enter

# # standard (int8)
#     # only_fc
#     echo "standard keyword int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # standard (int8)
#     # only_fc
#     echo "standard anomaly int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=standard TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

############################# standard_unroll int8 global #############################

make soft-tflm-clean

# # standard_unroll (int8)
#     # global
#     echo "standard_unroll imclass int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

# standard_unroll (int8)
    # global
    echo "standard_unroll mobilenet int8 global" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=global soft
    make fpga-program
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
    wait_enter

# # standard_unroll (int8)
#     # global
#     echo "standard_unroll keyword int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

# # standard_unroll (int8)
#     # global
#     echo "standard_unroll anomaly int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

############################# standard_unroll int8 kernel #############################

make soft-tflm-clean

# # standard_unroll (int8)
#     # kernel
#     echo "standard_unroll imclass int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# standard_unroll (int8)
    # kernel
    echo "standard_unroll mobilenet int8 kernel" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
    make fpga-program
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
    wait_enter

# # standard_unroll (int8)
#     # kernel
#     echo "standard_unroll keyword int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # standard_unroll (int8)
#     # kernel
#     echo "standard_unroll anomaly int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

############################# standard_unroll int8 only_cnn #############################

make soft-tflm-clean

# # standard_unroll (int8)
#     # only_cnn
#     echo "standard_unroll imclass int8 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# standard_unroll (int8)
    # only_cnn
    echo "standard_unroll mobilenet int8 only_cnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
    make fpga-program
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
    wait_enter

# # standard_unroll (int8)
#     # only_cnn
#     echo "standard_unroll keyword int8 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

############################# standard_unroll int8 only_dwcnn #############################

make soft-tflm-clean

# standard_unroll (int8)
    # only_dwcnn
    echo "standard_unroll mobilenet int8 only_dwcnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn soft
    make fpga-program
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn fpga-run
    wait_enter

# # standard_unroll (int8)
#     # only_dwcnn
#     echo "standard_unroll keyword int8 only_dwcnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn fpga-run
#     wait_enter

############################# standard_unroll int8 only_fc #############################

make soft-tflm-clean

# # standard_unroll (int8)
#     # only_fc
#     echo "standard_unroll imclass int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# standard_unroll (int8)
    # only_fc
    echo "standard_unroll mobilenet int8 only_fc" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
    make fpga-program
    make TFLM_TYPE=standard_unroll TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
    wait_enter

# # standard_unroll (int8)
#     # only_fc
#     echo "standard_unroll keyword int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # standard_unroll (int8)
#     # only_fc
#     echo "standard_unroll anomaly int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=standard_unroll TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=standard_unroll TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

############################# star_emulate int8 global #############################

make soft-tflm-clean

# # star_emulate (int8)
#     # global
#     echo "star_emulate imclass int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

# star_emulate (int8)
    # global
    echo "star_emulate mobilenet int8 global" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=global soft
    make fpga-program
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
    wait_enter

# # star_emulate (int8)
#     # global
#     echo "star_emulate keyword int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star_emulate (int8)
#     # global
#     echo "star_emulate anomaly int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

############################# star_emulate int8 kernel #############################

make soft-tflm-clean

# # star_emulate (int8)
#     # kernel
#     echo "star_emulate imclass int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# star_emulate (int8)
    # kernel
    echo "star_emulate mobilenet int8 kernel" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
    make fpga-program
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
    wait_enter

# # star_emulate (int8)
#     # kernel
#     echo "star_emulate keyword int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star_emulate (int8)
#     # kernel
#     echo "star_emulate anomaly int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

############################# star_emulate int8 only_cnn #############################

make soft-tflm-clean

# # star_emulate (int8)
#     # only_cnn
#     echo "star_emulate imclass int8 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# star_emulate (int8)
    # only_cnn
    echo "star_emulate mobilenet int8 only_cnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
    make fpga-program
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
    wait_enter

# # star_emulate (int8)
#     # only_cnn
#     echo "star_emulate keyword int8 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

############################# star_emulate int8 only_dwcnn #############################

make soft-tflm-clean

# star_emulate (int8)
    # only_dwcnn
    echo "star_emulate mobilenet int8 only_dwcnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn soft
    make fpga-program
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn fpga-run
    wait_enter

# # star_emulate (int8)
#     # only_dwcnn
#     echo "star_emulate keyword int8 only_dwcnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn fpga-run
#     wait_enter

############################# star_emulate int8 only_fc #############################

make soft-tflm-clean

# # star_emulate (int8)
#     # only_fc
#     echo "star_emulate imclass int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# star_emulate (int8)
    # only_fc
    echo "star_emulate mobilenet int8 only_fc" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
    make fpga-program
    make TFLM_TYPE=star_emulate TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
    wait_enter

# # star_emulate (int8)
#     # only_fc
#     echo "star_emulate keyword int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star_emulate (int8)
#     # only_fc
#     echo "star_emulate anomaly int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star_emulate TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star_emulate TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

############################ star global #############################

make soft-tflm-clean

# # star (int4)
#     # global
#     echo "star imclass int4 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (int8)
#     # global
#     echo "star imclass int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (int16)
#     # global
#     echo "star imclass int16 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (MPQ)
#     # global
#     echo "star imclass MPQ global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # global
#     echo "star imclass MPQ_UNPACK global measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=global fpga-run
#     wait_enter

# star (int4)
    # global
    echo "star mobilenet int4 global" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=global soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=global fpga-run
    wait_enter

# star (int8)
    # global
    echo "star mobilenet int8 global" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=global soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
    wait_enter

# star (int16)
    # global
    echo "star mobilenet int16 global" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=global soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=global fpga-run
    wait_enter

# star (MPQ)
    # global
    echo "star mobilenet MPQ global" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=global soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=global fpga-run
    wait_enter

# star (MPQ_UNPACK)
    # global
    echo "star mobilenet MPQ_UNPACK global measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=global soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=global fpga-run
    wait_enter

# # star (int4)
#     # global
#     echo "star keyword int4 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (int8)
#     # global
#     echo "star keyword int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (int16)
#     # global
#     echo "star keyword int16 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (MPQ)
#     # global
#     echo "star keyword MPQ global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # global
#     echo "star keyword MPQ_UNPACK global measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (int4)
#     # global
#     echo "star anomaly int4 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int4 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int4 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (int8)
#     # global
#     echo "star anomaly int8 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (int16)
#     # global
#     echo "star anomaly int16 global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int16 MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int16 MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (MPQ)
#     # global
#     echo "star anomaly MPQ global" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ MEASURE_TYPE=global fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # global
#     echo "star anomaly MPQ_UNPACK global measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=global soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=global fpga-run
#     wait_enter

############################ star kernel #############################

make soft-tflm-clean

# # star (int4)
#     # kernel
#     echo "star imclass int4 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (int8)
#     # kernel
#     echo "star imclass int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (int16)
#     # kernel
#     echo "star imclass int16 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (MPQ)
#     # kernel
#     echo "star imclass MPQ kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # kernel
#     echo "star imclass MPQ_UNPACK kernel measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=kernel fpga-run
#     wait_enter

# star (int4)
    # kernel
    echo "star mobilenet int4 kernel" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=kernel soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=kernel fpga-run
    wait_enter

# star (int8)
    # kernel
    echo "star mobilenet int8 kernel" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
    wait_enter

# star (int16)
    # kernel
    echo "star mobilenet int16 kernel" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=kernel soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=kernel fpga-run
    wait_enter

# star (MPQ)
    # kernel
    echo "star mobilenet MPQ kernel" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=kernel soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=kernel fpga-run
    wait_enter

# star (MPQ_UNPACK)
    # kernel
    echo "star mobilenet MPQ_UNPACK kernel measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=kernel soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=kernel fpga-run
    wait_enter

# # star (int4)
#     # kernel
#     echo "star keyword int4 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (int8)
#     # kernel
#     echo "star keyword int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (int16)
#     # kernel
#     echo "star keyword int16 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (MPQ)
#     # kernel
#     echo "star keyword MPQ kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # kernel
#     echo "star keyword MPQ_UNPACK kernel measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (int4)
#     # kernel
#     echo "star anomaly int4 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int4 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int4 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (int8)
#     # kernel
#     echo "star anomaly int8 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (int16)
#     # kernel
#     echo "star anomaly int16 kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int16 MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int16 MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (MPQ)
#     # kernel
#     echo "star anomaly MPQ kernel" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ MEASURE_TYPE=kernel fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # kernel
#     echo "star anomaly MPQ_UNPACK kernel measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=kernel soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=kernel fpga-run
#     wait_enter

############################ star only_cnn #############################

make soft-tflm-clean

# # star (int4)
#     # only_cnn
#     echo "star imclass int4 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# # star (int8)
#     # only_cnn
#     echo "star imclass int8 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# # star (int16)
#     # only_cnn
#     echo "star imclass int16 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# # star (MPQ)
#     # only_cnn
#     echo "star imclass MPQ only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # only_cnn
#     echo "star imclass MPQ_UNPACK only_cnn measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# star (int4)
    # only_cnn
    echo "star mobilenet int4 only_cnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=only_cnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=only_cnn fpga-run
    wait_enter

# star (int8)
    # only_cnn
    echo "star mobilenet int8 only_cnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
    wait_enter

# star (int16)
    # only_cnn
    echo "star mobilenet int16 only_cnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=only_cnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=only_cnn fpga-run
    wait_enter

# star (MPQ)
    # only_cnn
    echo "star mobilenet MPQ only_cnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=only_cnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=only_cnn fpga-run
    wait_enter

# star (MPQ_UNPACK)
    # only_cnn
    echo "star mobilenet MPQ_UNPACK only_cnn measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_cnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_cnn fpga-run
    wait_enter

# # star (int4)
#     # only_cnn
#     echo "star keyword int4 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# # star (int8)
#     # only_cnn
#     echo "star keyword int8 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# # star (int16)
#     # only_cnn
#     echo "star keyword int16 only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# # star (MPQ)
#     # only_cnn
#     echo "star keyword MPQ only_cnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # only_cnn
#     echo "star keyword MPQ_UNPACK only_cnn measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_cnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_cnn fpga-run
#     wait_enter

############################ star only_dwcnn #############################

make soft-tflm-clean

# star (int4)
    # only_dwcnn
    echo "star mobilenet int4 only_dwcnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=only_dwcnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=only_dwcnn fpga-run
    wait_enter

# star (int8)
    # only_dwcnn
    echo "star mobilenet int8 only_dwcnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn fpga-run
    wait_enter

# star (int16)
    # only_dwcnn
    echo "star mobilenet int16 only_dwcnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=only_dwcnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=only_dwcnn fpga-run
    wait_enter

# star (MPQ)
    # only_dwcnn
    echo "star mobilenet MPQ only_dwcnn" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=only_dwcnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=only_dwcnn fpga-run
    wait_enter

# star (MPQ_UNPACK)
    # only_dwcnn
    echo "star mobilenet MPQ_UNPACK only_dwcnn measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_dwcnn soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_dwcnn fpga-run
    wait_enter

# # star (int4)
#     # only_dwcnn
#     echo "star keyword int4 only_dwcnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=only_dwcnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=only_dwcnn fpga-run
#     wait_enter

# # star (int8)
#     # only_dwcnn
#     echo "star keyword int8 only_dwcnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_dwcnn fpga-run
#     wait_enter

# # star (int16)
#     # only_dwcnn
#     echo "star keyword int16 only_dwcnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=only_dwcnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=only_dwcnn fpga-run
#     wait_enter

# # star (MPQ)
#     # only_dwcnn
#     echo "star keyword MPQ only_dwcnn" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=only_dwcnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=only_dwcnn fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # only_dwcnn
#     echo "star keyword MPQ_UNPACK only_dwcnn measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_dwcnn soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_dwcnn fpga-run
#     wait_enter

############################ star only_fc #############################

make soft-tflm-clean

# # star (int4)
#     # only_fc
#     echo "star imclass int4 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (int8)
#     # only_fc
#     echo "star imclass int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (int16)
#     # only_fc
#     echo "star imclass int16 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (MPQ)
#     # only_fc
#     echo "star imclass MPQ only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # only_fc
#     echo "star imclass MPQ_UNPACK only_fc measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# star (int4)
    # only_fc
    echo "star mobilenet int4 only_fc" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=only_fc soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=only_fc fpga-run
    wait_enter

# star (int8)
    # only_fc
    echo "star mobilenet int8 only_fc" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
    wait_enter

# star (int16)
    # only_fc
    echo "star mobilenet int16 only_fc" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=only_fc soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=only_fc fpga-run
    wait_enter

# star (MPQ)
    # only_fc
    echo "star mobilenet MPQ only_fc" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=only_fc soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=only_fc fpga-run
    wait_enter

# star (MPQ_UNPACK)
    # only_fc
    echo "star mobilenet MPQ_UNPACK only_fc measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_fc soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_fc fpga-run
    wait_enter

# # star (int4)
#     # only_fc
#     echo "star keyword int4 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (int8)
#     # only_fc
#     echo "star keyword int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (int16)
#     # only_fc
#     echo "star keyword int16 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (MPQ)
#     # only_fc
#     echo "star keyword MPQ only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # only_fc
#     echo "star keyword MPQ_UNPACK only_fc measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (int4)
#     # only_fc
#     echo "star anomaly int4 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int4 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int4 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (int8)
#     # only_fc
#     echo "star anomaly int8 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (int16)
#     # only_fc
#     echo "star anomaly int16 only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int16 MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int16 MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (MPQ)
#     # only_fc
#     echo "star anomaly MPQ only_fc" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ MEASURE_TYPE=only_fc fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # only_fc
#     echo "star anomaly MPQ_UNPACK only_fc measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_fc soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=only_fc fpga-run
#     wait_enter

############################ star unpack #############################

make soft-tflm-clean

# # star (int4)
#     # unpacking
#     echo "star imclass int4 unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int4 MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (int8)
#     # unpacking
#     echo "star imclass int8 unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int8 MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (int16)
#     # unpacking
#     echo "star imclass int16 unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=int16 MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (MPQ)
#     # unpacking
#     echo "star imclass MPQ unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # unpacking
#     echo "star imclass MPQ_UNPACK unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=imclass QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# star (int4)
    # unpacking
    echo "star mobilenet int4 unpacking measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=unpacking soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int4 MEASURE_TYPE=unpacking fpga-run
    wait_enter

# star (int8)
    # unpacking
    echo "star mobilenet int8 unpacking measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=unpacking soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int8 MEASURE_TYPE=unpacking fpga-run
    wait_enter

# star (int16)
    # unpacking
    echo "star mobilenet int16 unpacking measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=unpacking soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=int16 MEASURE_TYPE=unpacking fpga-run
    wait_enter

# star (MPQ)
    # unpacking
    echo "star mobilenet MPQ unpacking measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=unpacking soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ MEASURE_TYPE=unpacking fpga-run
    wait_enter

# star (MPQ_UNPACK)
    # unpacking
    echo "star mobilenet MPQ_UNPACK unpacking measure" >> /home/edward.manca/prova.txt
    make profpga-close-fpga
    make clean
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=unpacking soft
    make fpga-program
    make TFLM_TYPE=star TINY_NET=mobilenet QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=unpacking fpga-run
    wait_enter

# # star (int4)
#     # unpacking
#     echo "star keyword int4 unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int4 MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (int8)
#     # unpacking
#     echo "star keyword int8 unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int8 MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (int16)
#     # unpacking
#     echo "star keyword int16 unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=int16 MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (MPQ)
#     # unpacking
#     echo "star keyword MPQ unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # unpacking
#     echo "star keyword MPQ_UNPACK unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=keyword QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (int4)
#     # unpacking
#     echo "star anomaly int4 unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int4 MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int4 MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (int8)
#     # unpacking
#     echo "star anomaly int8 unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int8 MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (int16)
#     # unpacking
#     echo "star anomaly int16 unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int16 MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=int16 MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (MPQ)
#     # unpacking
#     echo "star anomaly MPQ unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ MEASURE_TYPE=unpacking fpga-run
#     wait_enter

# # star (MPQ_UNPACK)
#     # unpacking
#     echo "star anomaly MPQ_UNPACK unpacking measure" >> /home/edward.manca/prova.txt
#     make profpga-close-fpga
#     make clean
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=unpacking soft
#     make fpga-program
#     make TFLM_TYPE=star TINY_NET=anomaly QUANT_TYPE=MPQ_UNPACK MEASURE_TYPE=unpacking fpga-run
#     wait_enter


############### SWITCH OFF FPGA ################
make profpga-close-fpga