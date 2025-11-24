from tqdm import tqdm
import numpy as np

CHUNK_SIZE = 10000

def run_simulation(fp, NT, dset_shape, dset_chunks, initialize_func, step_chunk, get_I_ext_chunk):
    
    initialize_func()

    dset = fp.create_dataset(
        "vars",
        shape=dset_shape,
        dtype="float64",
        chunks=dset_chunks,
    )
    
    chunk_template = np.empty(dset_chunks, dtype=np.float64)

    chunk_num = NT // CHUNK_SIZE

    for cn in tqdm(range(chunk_num)):
        start_index = CHUNK_SIZE * cn
        end_index = start_index + CHUNK_SIZE
        
        chunk = chunk_template.copy()
        i_ext_chunk = get_I_ext_chunk(start_index, end_index)
        step_chunk(chunk, i_ext_chunk)
        dset[start_index:end_index] = chunk

    remaining_steps = NT % CHUNK_SIZE
    if remaining_steps > 0:
        start_index = CHUNK_SIZE * chunk_num
        end_index = start_index + remaining_steps
        
        remaining_chunk = chunk_template[:remaining_steps].copy()
        i_ext_chunk = get_I_ext_chunk(start_index, end_index)
        step_chunk(remaining_chunk, i_ext_chunk)
        dset[start_index:end_index] = remaining_chunk
