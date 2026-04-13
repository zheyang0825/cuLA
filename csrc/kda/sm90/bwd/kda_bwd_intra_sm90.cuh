#pragma once

#include "kda_bwd_common.h"

namespace sm90_bwd {

void
run_kda_bwd_intra_sm90(KDA_bwd_intra_params& params, cudaStream_t stream);

}  // namespace sm90_bwd
