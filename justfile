list:
    just --list
update-torch-nightly:
    pip install -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117
    pip install -U --pre triton
update-torch-cpp:
    #!/usr/bin/env bash
    cd third_party
    wget https://download.pytorch.org/libtorch/nightly/cu117/libtorch-cxx11-abi-shared-with-deps-latest.zip
    unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
build-cpp:
    cd build; make -j