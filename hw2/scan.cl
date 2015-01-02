
__kernel void scan_blelloch(__global float * a, __global float * r, __global float * aux, __local float * b, int n)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint dp = 1;
    bool valid = gid < n;
    b[lid] = valid ? a[gid] : 0;
    
    for(uint s = block_size>>1; s > 0; s >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < s)
        {
            uint i = dp*(2*lid+1)-1;
            uint j = dp*(2*lid+2)-1;
            b[j] += b[i];
        }
        
        dp <<= 1;
    }
    
    if(lid == 0) {
        uint group_id = get_group_id(0);
        aux[group_id] = b[block_size - 1];	
        if (valid) {
            uint last_idx = group_id * block_size + block_size - 1;
            r[last_idx] = b[block_size - 1];	
        }
        b[block_size - 1] = 0;
    }
    
    
    for(uint s = 1; s < block_size; s <<= 1)
    {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if(lid < s)
        {
            uint i = dp*(2*lid+1)-1;
            uint j = dp*(2*lid+2)-1;
            
            float t = b[j];
            b[j] += b[i];
            b[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < block_size - 1 && valid)
        r[gid] = b[lid + 1];
    
}


__kernel void add_aux(__global int * aux, __global int * out, int n)
{
    uint gid = get_global_id(0);
    uint group_id = get_group_id(0);
    if (gid > n - 1 || group_id == 0) return;
    out[gid] += aux[group_id - 1]; 
}
