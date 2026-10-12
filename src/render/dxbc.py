"""Vectorized CPU execution of installed DXBC shader assembly.

Registers retain IEEE float32/uint32 bits, including integer Bayer operations.
This is an instruction executor, not a substitute AIT lighting model. Resource
sampling is supplied explicitly by the caller; missing textures, constants,
unsupported instructions and unimplemented sampler modes are errors.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np

class UnsupportedShader(ValueError):
    pass

def _arguments(line):
    # Literal vector arguments contain commas; register/index operands do not.
    return [s.strip() for s in re.split(r',\s*(?![^()]*\))', line)]

def _bits(value):
    return np.asarray(value,dtype=np.float32).view(np.uint32)

@dataclass
class ShaderResult:
    outputs: dict
    discarded: np.ndarray
    instruction_count: int

class Program:
    def __init__(self, assembly):
        self.instructions=[]
        self.stage=None
        self.constants={}
        for raw in assembly.splitlines():
            line=raw.strip()
            if not line or line.startswith('//'):
                continue
            if re.fullmatch(r'[pv]s_\d_\d',line):
                self.stage=line[:2]
                continue
            if line.startswith('dcl_constantbuffer '):
                match=re.search(r'CB(\d+)\[(\d+)\]',line)
                self.constants[int(match[1])]=int(match[2])
            if line.startswith('dcl_'):
                continue
            command,*rest=line.split(None,1)
            self.instructions.append((command,_arguments(rest[0]) if rest else []))
        if self.stage is None:
            raise UnsupportedShader('Missing supported shader model signature')

    def execute(self, inputs, constant_buffers, *, sample=None):
        inputs={k:np.asarray(v,np.float32) for k,v in inputs.items()}
        count=next(iter(inputs.values())).shape[0]
        if any(v.shape!=(count,4) for v in inputs.values()):
            raise ValueError('Each input register must be N x 4 float32')
        regs={k:v.view(np.uint32).copy() for k,v in inputs.items()}
        constants={}
        for index,length in self.constants.items():
            if index not in constant_buffers:
                raise UnsupportedShader(f'Missing actual constant buffer cb{index}')
            value=np.asarray(constant_buffers[index])
            if value.ndim!=2 or value.shape[1]!=4 or value.shape[0]<length:
                raise ValueError(f'cb{index} requires at least {length} float4 vectors')
            constants[index]=value.copy() if value.dtype==np.uint32 else _bits(value)
        active=np.ones(count,bool)
        discarded=np.zeros(count,bool)
        stack=[]
        loops=[]

        def operand(text):
            negate=text.startswith('-')
            if negate:
                text=text[1:]
            absolute=text.startswith('|') and text.endswith('|')
            if absolute:
                text=text[1:-1]
            if text.startswith('l('):
                values=[]
                for token in _arguments(text[2:-1]):
                    if '.' in token or 'e' in token.lower():
                        values.append(_bits(float(token)).item())
                    else:
                        values.append(int(token,0)&0xffffffff)
                values=np.array(values,np.uint32)
                if len(values)==1:
                    values=np.repeat(values,4)
                if len(values)!=4:
                    raise UnsupportedShader('Unsupported literal size: '+text)
                result=np.broadcast_to(values,(count,4)).copy()
            else:
                match=re.fullmatch(r'(cb(\d+)\[(\d+)\]|[rvo]\d+)(?:\.([xyzw]{1,4}))?',text)
                if not match:
                    raise UnsupportedShader('Unsupported operand: '+text)
                name,buffer,index,swizzle=match.groups()
                if buffer is not None:
                    result=np.broadcast_to(constants[int(buffer)][int(index)],(count,4))
                else:
                    if name not in regs:
                        raise UnsupportedShader('Unbound register: '+name)
                    result=regs[name]
                if swizzle:
                    if len(swizzle)==1:
                        swizzle*=4
                    elif len(swizzle)!=4:
                        # Assembly operands normally have full four-lane
                        # swizzles; shorter input vectors repeat their last lane.
                        swizzle+=swizzle[-1]*(4-len(swizzle))
                    result=result[:,['xyzw'.index(c) for c in swizzle]]
            if absolute or negate:
                result=result.view(np.float32).copy()
                if absolute:
                    result=np.abs(result)
                if negate:
                    result=-result
                result=result.view(np.uint32)
            return result

        def put(destination, value, saturate=False):
            if destination=='null':
                return
            match=re.fullmatch(r'([ro]\d+)(?:\.([xyzw]{1,4}))?',destination)
            if not match:
                raise UnsupportedShader('Unsupported destination '+destination)
            name,mask=match.groups()
            lanes=['xyzw'.index(c) for c in mask or 'xyzw']
            if saturate:
                value=np.clip(value.view(np.float32),0,1).view(np.uint32)
            if name not in regs:
                regs[name]=np.zeros((count,4),np.uint32)
            target=regs[name]
            rows=np.flatnonzero(active)
            target[np.ix_(rows,lanes)]=value[np.ix_(rows,lanes)]

        line=0
        executed=0
        def loop_mask():
            return loops[-1]['running'] if loops else np.ones(count,bool)
        while line<len(self.instructions):
            opcode,args=self.instructions[line]
            line+=1
            executed+=1
            if executed>1000000:
                raise UnsupportedShader('Shader control flow exceeded execution safety limit')
            if opcode=='loop':
                loops.append({'start':line,'entry':active.copy(),'running':active.copy()})
                continue
            if opcode in ('breakc_nz','breakc_z','break'):
                if not loops:
                    raise UnsupportedShader('Break outside loop')
                condition=np.ones(count,bool) if opcode=='break' else operand(args[0])[:,0]!=0
                if opcode=='breakc_z':
                    condition=~condition
                loops[-1]['running'] &= ~(active & condition)
                active &= loops[-1]['running']
                continue
            if opcode=='endloop':
                if not loops:
                    raise UnsupportedShader('Endloop outside loop')
                loops[-1]['running'] &= ~discarded
                if loops[-1]['running'].any():
                    active=loops[-1]['running'].copy()
                    line=loops[-1]['start']
                else:
                    active=loops.pop()['entry'] & ~discarded & loop_mask()
                continue
            if opcode in ('if_nz','if_z'):
                condition=operand(args[0])[:,0]!=0
                if opcode=='if_z':
                    condition=~condition
                stack.append((active.copy(),condition))
                active &= condition
                continue
            if opcode=='else':
                parent,condition=stack[-1]
                active=parent & ~condition & ~discarded & loop_mask()
                continue
            if opcode=='endif':
                active=stack.pop()[0] & ~discarded & loop_mask()
                continue
            if opcode.startswith('discard_'):
                condition=operand(args[0])[:,0]!=0
                if opcode=='discard_z':
                    condition=~condition
                discarded |= active & condition
                active &= ~discarded
                continue
            if opcode=='ret':
                if stack:
                    raise UnsupportedShader('Conditional return not implemented')
                break
            if not active.any():
                continue
            saturation=opcode.endswith('_sat')
            command=opcode[:-4] if saturation else opcode
            if command=='sincos':
                value=operand(args[2]).view(np.float32)
                put(args[0],_bits(np.sin(value)),saturation)
                put(args[1],_bits(np.cos(value)),saturation)
                continue
            if command=='imul':
                a=operand(args[2]).view(np.int32).astype(np.int64)
                b=operand(args[3]).view(np.int32).astype(np.int64)
                product=a*b
                put(args[0],(product>>32).astype(np.uint32))
                put(args[1],product.astype(np.uint32))
                continue
            if command.startswith('sample'):
                if sample is None:
                    raise UnsupportedShader('Texture sampling callback required')
                resource=re.fullmatch(r't(\d+)(?:\.([xyzw]{4}))?',args[2])
                sampler=re.fullmatch(r's(\d+)',args[3])
                if not resource or not sampler:
                    raise UnsupportedShader('Unrecognized resource operand')
                extra=operand(args[4]).view(np.float32)[:,0] if len(args)>4 else None
                result=np.asarray(sample(command,int(resource[1]),int(sampler[1]),
                    operand(args[1]).view(np.float32),extra),np.float32)
                if result.shape!=(count,4):
                    raise ValueError('Sampler must return N x 4 float32')
                if resource[2]:
                    result=result[:,['xyzw'.index(c) for c in resource[2]]]
                put(args[0],result.view(np.uint32),saturation)
                continue
            values=[operand(a) for a in args[1:]]
            f=[v.view(np.float32) for v in values]
            with np.errstate(all='ignore'):
                if command=='mov': result=values[0]
                elif command=='movc': result=np.where(values[0]!=0,values[1],values[2])
                elif command=='add': result=_bits(f[0]+f[1])
                elif command=='mul': result=_bits(f[0]*f[1])
                elif command=='mad': result=_bits(f[0]*f[1]+f[2])
                elif command=='div': result=_bits(f[0]/f[1])
                elif command in ('dp2','dp3','dp4'):
                    n=int(command[-1]); result=_bits(np.repeat(np.sum(f[0][:,:n]*f[1][:,:n],axis=1)[:,None],4,axis=1))
                elif command=='min': result=_bits(np.fmin(f[0],f[1]))
                elif command=='max': result=_bits(np.fmax(f[0],f[1]))
                elif command=='sqrt': result=_bits(np.sqrt(f[0]))
                elif command=='rsq': result=_bits(1/np.sqrt(f[0]))
                elif command=='log': result=_bits(np.log2(f[0]))
                elif command=='exp': result=_bits(np.exp2(f[0]))
                elif command=='frc': result=_bits(f[0]-np.floor(f[0]))
                elif command in ('ge','lt','eq','ne'):
                    fn={'ge':np.greater_equal,'lt':np.less,'eq':np.equal,'ne':np.not_equal}[command]
                    result=np.where(fn(f[0],f[1]),np.uint32(0xffffffff),np.uint32(0))
                elif command in ('ige','ilt','ieq','ine'):
                    fn={'ige':np.greater_equal,'ilt':np.less,'ieq':np.equal,'ine':np.not_equal}[command]
                    result=np.where(fn(values[0].view(np.int32),values[1].view(np.int32)),np.uint32(0xffffffff),np.uint32(0))
                elif command=='and': result=values[0]&values[1]
                elif command=='or': result=values[0]|values[1]
                elif command=='ishl': result=values[0] << (values[1]&31)
                elif command=='iadd': result=(values[0].view(np.int32)+values[1].view(np.int32)).view(np.uint32)
                elif command=='ftoi': result=np.trunc(f[0]).astype(np.int32).view(np.uint32)
                elif command=='itof': result=_bits(values[0].view(np.int32).astype(np.float32))
                elif command=='utof': result=_bits(values[0].astype(np.float32))
                else: raise UnsupportedShader(f'Instruction {line}: {opcode}')
            put(args[0],result,saturation)
        if stack or loops:
            raise UnsupportedShader('Unbalanced shader control flow')
        return ShaderResult({k:v.view(np.float32) for k,v in regs.items() if k.startswith('o')},
                            discarded,executed)
