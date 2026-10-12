"""Bind compiled Unity constant layouts without guessed defaults.

Uniform values must already use GPU encodings (linear colors where required,
actual shader-global vectors and column-major matrices). This module does not
infer a light rig, SH probe or texture when capture data is absent.
"""
from __future__ import annotations
import numpy as np
from .dxbc import Program, UnsupportedShader

def constant_buffers(record, program: Program, uniforms):
    names={index:name for name,index in record['name_indices']}
    bindings=record['bindings']
    slots={row['m_NameIndex']:row['m_Index'] for row in bindings['m_ConstantBufferBindings']}
    result={}
    for cb in bindings['m_ConstantBuffers']:
        slot=slots[cb['m_NameIndex']]
        if slot not in program.constants:
            continue
        if cb.get('m_StructParams'):
            raise UnsupportedShader('Structured constant buffer layout not implemented')
        # Padding is not a shader value. Only named declared parameters may be
        # accessed; all of those must be supplied by captured/recovered state.
        buffer=np.zeros((max(program.constants[slot],(cb['m_Size']+15)//16),4),np.uint32)
        flat=buffer.reshape(-1)
        for field in cb['m_VectorParams']+cb['m_MatrixParams']:
            name=names[field['m_NameIndex']]
            if name not in uniforms:
                raise UnsupportedShader('Missing actual GPU uniform '+name)
            count=field['m_ArraySize'] or 1
            width=field.get('m_Dim',field.get('m_RowCount',0)*4)
            expected=count*width
            values=np.asarray(uniforms[name])
            # Matrix representation is deliberately explicit to avoid hidden
            # transposes. Caller supplies flattened GPU constant-memory order.
            if values.size!=expected:
                raise ValueError(f'{name}: expected {expected} components, got {values.size}')
            typ=field['m_Type']
            if typ==0:
                bits=values.astype(np.float32).reshape(-1).view(np.uint32)
            elif typ==1:
                bits=values.astype(np.int32).reshape(-1).view(np.uint32)
            elif typ==2:
                bits=values.astype(np.bool_).astype(np.uint32).reshape(-1)
            else:
                raise UnsupportedShader(f'{name}: unsupported constant type {typ}')
            start=field['m_Index']//4
            if field['m_Index']%4:
                raise UnsupportedShader('Non-aligned constant field')
            # Array elements are float4-aligned, even for scalar/vector arrays.
            stride=4*((width+3)//4) if count>1 else width
            for element in range(count):
                first=start+element*stride
                if first+width>len(flat):
                    raise ValueError('Compiled field extends beyond declared cbuffer')
                flat[first:first+width]=bits[element*width:(element+1)*width]
        result[slot]=buffer
    if set(result)!=set(program.constants):
        raise UnsupportedShader('Incomplete compiled constant buffer bindings')
    return result

def select_program(shader, pass_index, stage, required_keywords):
    """Select only an exact compiled keyword branch; no nearest-variant fallback."""
    keywords=set(required_keywords)
    rows=[r for p in shader['passes'] if p['pass']==pass_index for r in p['programs']
          if r['stage']==stage and set(r['keywords'])==keywords]
    if len(rows)!=1:
        raise UnsupportedShader(f'Exactly one {stage} branch required; got {len(rows)} for {sorted(keywords)}')
    return rows[0]
