"""Original DXGI byte storage and D3D11 texture transfer intervals.

No pixel decoding, gamma conversion, generated mips or unknown-byte fill. Pitches
follow pinned RenderDoc MapIntercept/Serialise_CreateTextureData and DXGI enums.
"""
from math import ceil

class UnresolvedTexture(RuntimeError):pass

def block_layout(fmt):
    if 70<=fmt<=84 or 94<=fmt<=99:return 4,4,8 if fmt in (*range(70,73),*range(79,82)) else 16
    if fmt==66:return 8,1,1  # R1_UNORM, not R9G9B9E5_SHAREDEXP.
    if fmt in (68,69):return 2,1,4
    if 1<=fmt<=4:n=16
    elif 5<=fmt<=8:n=12
    elif 9<=fmt<=22:n=8
    elif 23<=fmt<=47 or fmt==67 or 87<=fmt<=93:n=4
    elif 48<=fmt<=59 or fmt in (85,86):n=2
    elif 60<=fmt<=65:n=1
    else:raise UnresolvedTexture('DXGI storage format not implemented: '+str(fmt))
    return 1,1,n

def layouts(kind,desc):
    width=desc['Width'];height=desc.get('Height',1);depth=desc.get('Depth',1)
    mips=desc['MipLevels'] or max(width,height,depth).bit_length()
    arrays=1 if kind=='CreateTexture3D' else desc['ArraySize']
    bw,bh,size=block_layout(desc['Format']);result=[]
    for array in range(arrays):
        for mip in range(mips):
            w=max(1,width>>mip);h=max(1,height>>mip);d=max(1,depth>>mip)
            columns=ceil(w/bw);rows=ceil(h/bh);pitch=columns*size;slice_pitch=pitch*rows
            result.append({'subresource':mip+array*mips,'array':array,'mip':mip,'width':w,'height':h,'depth':d,
                'block_width':bw,'block_height':bh,'block_bytes':size,'columns':columns,'rows':rows,
                'row_pitch':pitch,'slice_pitch':slice_pitch,'bytes':slice_pitch*d})
    return result

def paint(segments,start,payload,total,*,discard=False):
    end=start+len(payload)
    if start<0 or end>total:raise UnresolvedTexture('Texture write is outside captured subresource')
    result=[]
    for offset,data in ([] if discard else segments):
        stop=offset+len(data)
        if stop<=start or offset>=end:result.append((offset,data));continue
        if offset<start:result.append((offset,data[:start-offset]))
        if stop>end:result.append((end,data[end-offset:]))
    if payload:result.append((start,payload))
    return sorted(result)

def box_region(layout,box):
    if box is None:box={'left':0,'top':0,'front':0,'right':layout['width'],'bottom':layout['height'],'back':layout['depth']}
    x,y,z=box['left'],box['top'],box['front'];r,b,k=box['right'],box['bottom'],box['back']
    if not (0<=x<=r<=layout['width'] and 0<=y<=b<=layout['height'] and 0<=z<=k<=layout['depth']):raise UnresolvedTexture('Native update box outside subresource')
    if x==r or y==b or z==k:return None
    bw,bh=layout['block_width'],layout['block_height']
    if x%bw or y%bh or (r%bw and r!=layout['width']) or (b%bh and b!=layout['height']):
        raise UnresolvedTexture('Unaligned encoded-block/bit texture box must not be approximated')
    return x//bw,y//bh,z,ceil(r/bw)-x//bw,ceil(b/bh)-y//bh,k-z

def update_box(segments,layout,box,payload,row_pitch,depth_pitch,*,discard=False):
    region=box_region(layout,box)
    if region is None:return segments
    x,y,z,columns,rows,slices=region;row_bytes=columns*layout['block_bytes']
    if row_pitch<row_bytes or (slices>1 and depth_pitch<row_pitch*rows):raise UnresolvedTexture('Recorded source pitches too small')
    needed=(slices-1)*depth_pitch+(rows-1)*row_pitch+row_bytes
    if len(payload)!=needed:raise UnresolvedTexture('SourceDataLength does not match serialized pitched box data')
    result=[] if discard else segments
    for dz in range(slices):
        for dy in range(rows):
            src=dz*depth_pitch+dy*row_pitch
            dst=(z+dz)*layout['slice_pitch']+(y+dy)*layout['row_pitch']+x*layout['block_bytes']
            result=paint(result,dst,payload[src:src+row_bytes],layout['bytes'])
    return result
