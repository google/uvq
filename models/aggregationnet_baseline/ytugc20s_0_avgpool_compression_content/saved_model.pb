“К
ЏЊ
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ъ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Ј—8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
9
VarIsInitializedOp
resource
is_initialized
И"serve*2.5.02unknown8ЄЌ

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
К
global_stepVarHandleOp*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	
Ш
PlaceholderPlaceholder*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d*
dtype0*1
shape(:&€€€€€€€€€€€€€€€€€€d
Ъ
Placeholder_1Placeholder*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d*
dtype0*1
shape(:&€€€€€€€€€€€€€€€€€€d
Ъ
Placeholder_2Placeholder*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d*
dtype0*1
shape(:&€€€€€€€€€€€€€€€€€€d
s
feature_compressionIdentityPlaceholder*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d
t
feature_distortionIdentityPlaceholder_1*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d
q
feature_contentIdentityPlaceholder_2*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
Ц
concatConcatV2feature_compressionfeature_contentconcat/axis*
N*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€»
c
featureIdentityconcat*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€»
S
Model/time_distributed/ShapeShapefeature*
T0*
_output_shapes
:
t
*Model/time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
v
,Model/time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
v
,Model/time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
†
$Model/time_distributed/strided_sliceStridedSliceModel/time_distributed/Shape*Model/time_distributed/strided_slice/stack,Model/time_distributed/strided_slice/stack_1,Model/time_distributed/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
}
$Model/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      »   
У
Model/time_distributed/ReshapeReshapefeature$Model/time_distributed/Reshape/shape*
T0*0
_output_shapes
:€€€€€€€€€»
ч
UModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/shapeConst*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
:*
dtype0*%
valueB"      »      
б
SModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/minConst*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: *
dtype0*
valueB
 *хлкљ
б
SModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/maxConst*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: *
dtype0*
valueB
 *хлк=
ќ
]Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/RandomUniformRandomUniformUModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/shape*
T0*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*(
_output_shapes
:»А*
dtype0
о
SModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/subSubSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/maxSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/min*
T0*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: 
К
SModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/mulMul]Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/RandomUniformSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/sub*
T0*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*(
_output_shapes
:»А
ь
OModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniformAddSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/mulSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/min*
T0*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*(
_output_shapes
:»А
Ч
4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelVarHandleOp*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: *
dtype0*
shape:»А*E
shared_name64Model/time_distributed/head2d/Head/Conv2D_1x1/kernel
є
UModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp4Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: 
г
;Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/AssignAssignVariableOp4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelOModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform*
dtype0
«
HModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Read/ReadVariableOpReadVariableOp4Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*(
_output_shapes
:»А*
dtype0
Џ
DModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Initializer/zerosConst*E
_class;
97loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/bias*
_output_shapes	
:А*
dtype0*
valueBА*    
Д
2Model/time_distributed/head2d/Head/Conv2D_1x1/biasVarHandleOp*E
_class;
97loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/bias*
_output_shapes
: *
dtype0*
shape:А*C
shared_name42Model/time_distributed/head2d/Head/Conv2D_1x1/bias
µ
SModel/time_distributed/head2d/Head/Conv2D_1x1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp2Model/time_distributed/head2d/Head/Conv2D_1x1/bias*
_output_shapes
: 
‘
9Model/time_distributed/head2d/Head/Conv2D_1x1/bias/AssignAssignVariableOp2Model/time_distributed/head2d/Head/Conv2D_1x1/biasDModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Initializer/zeros*
dtype0
ґ
FModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Read/ReadVariableOpReadVariableOp2Model/time_distributed/head2d/Head/Conv2D_1x1/bias*
_output_shapes	
:А*
dtype0
¬
CModel/time_distributed/head2d/Head/Conv2D_1x1/Conv2D/ReadVariableOpReadVariableOp4Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*(
_output_shapes
:»А*
dtype0
Ж
4Model/time_distributed/head2d/Head/Conv2D_1x1/Conv2DConv2DModel/time_distributed/ReshapeCModel/time_distributed/head2d/Head/Conv2D_1x1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides

і
DModel/time_distributed/head2d/Head/Conv2D_1x1/BiasAdd/ReadVariableOpReadVariableOp2Model/time_distributed/head2d/Head/Conv2D_1x1/bias*
_output_shapes	
:А*
dtype0
ч
5Model/time_distributed/head2d/Head/Conv2D_1x1/BiasAddBiasAdd4Model/time_distributed/head2d/Head/Conv2D_1x1/Conv2DDModel/time_distributed/head2d/Head/Conv2D_1x1/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А
Ћ
<Model/time_distributed/head2d/Head/BN/gamma/Initializer/onesConst*>
_class4
20loc:@Model/time_distributed/head2d/Head/BN/gamma*
_output_shapes	
:А*
dtype0*
valueBА*  А?
п
+Model/time_distributed/head2d/Head/BN/gammaVarHandleOp*>
_class4
20loc:@Model/time_distributed/head2d/Head/BN/gamma*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Model/time_distributed/head2d/Head/BN/gamma
І
LModel/time_distributed/head2d/Head/BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp+Model/time_distributed/head2d/Head/BN/gamma*
_output_shapes
: 
Њ
2Model/time_distributed/head2d/Head/BN/gamma/AssignAssignVariableOp+Model/time_distributed/head2d/Head/BN/gamma<Model/time_distributed/head2d/Head/BN/gamma/Initializer/ones*
dtype0
®
?Model/time_distributed/head2d/Head/BN/gamma/Read/ReadVariableOpReadVariableOp+Model/time_distributed/head2d/Head/BN/gamma*
_output_shapes	
:А*
dtype0
 
<Model/time_distributed/head2d/Head/BN/beta/Initializer/zerosConst*=
_class3
1/loc:@Model/time_distributed/head2d/Head/BN/beta*
_output_shapes	
:А*
dtype0*
valueBА*    
м
*Model/time_distributed/head2d/Head/BN/betaVarHandleOp*=
_class3
1/loc:@Model/time_distributed/head2d/Head/BN/beta*
_output_shapes
: *
dtype0*
shape:А*;
shared_name,*Model/time_distributed/head2d/Head/BN/beta
•
KModel/time_distributed/head2d/Head/BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp*Model/time_distributed/head2d/Head/BN/beta*
_output_shapes
: 
Љ
1Model/time_distributed/head2d/Head/BN/beta/AssignAssignVariableOp*Model/time_distributed/head2d/Head/BN/beta<Model/time_distributed/head2d/Head/BN/beta/Initializer/zeros*
dtype0
¶
>Model/time_distributed/head2d/Head/BN/beta/Read/ReadVariableOpReadVariableOp*Model/time_distributed/head2d/Head/BN/beta*
_output_shapes	
:А*
dtype0
Ў
CModel/time_distributed/head2d/Head/BN/moving_mean/Initializer/zerosConst*D
_class:
86loc:@Model/time_distributed/head2d/Head/BN/moving_mean*
_output_shapes	
:А*
dtype0*
valueBА*    
Б
1Model/time_distributed/head2d/Head/BN/moving_meanVarHandleOp*D
_class:
86loc:@Model/time_distributed/head2d/Head/BN/moving_mean*
_output_shapes
: *
dtype0*
shape:А*B
shared_name31Model/time_distributed/head2d/Head/BN/moving_mean
≥
RModel/time_distributed/head2d/Head/BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp1Model/time_distributed/head2d/Head/BN/moving_mean*
_output_shapes
: 
—
8Model/time_distributed/head2d/Head/BN/moving_mean/AssignAssignVariableOp1Model/time_distributed/head2d/Head/BN/moving_meanCModel/time_distributed/head2d/Head/BN/moving_mean/Initializer/zeros*
dtype0
і
EModel/time_distributed/head2d/Head/BN/moving_mean/Read/ReadVariableOpReadVariableOp1Model/time_distributed/head2d/Head/BN/moving_mean*
_output_shapes	
:А*
dtype0
я
FModel/time_distributed/head2d/Head/BN/moving_variance/Initializer/onesConst*H
_class>
<:loc:@Model/time_distributed/head2d/Head/BN/moving_variance*
_output_shapes	
:А*
dtype0*
valueBА*  А?
Н
5Model/time_distributed/head2d/Head/BN/moving_varianceVarHandleOp*H
_class>
<:loc:@Model/time_distributed/head2d/Head/BN/moving_variance*
_output_shapes
: *
dtype0*
shape:А*F
shared_name75Model/time_distributed/head2d/Head/BN/moving_variance
ї
VModel/time_distributed/head2d/Head/BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp5Model/time_distributed/head2d/Head/BN/moving_variance*
_output_shapes
: 
№
<Model/time_distributed/head2d/Head/BN/moving_variance/AssignAssignVariableOp5Model/time_distributed/head2d/Head/BN/moving_varianceFModel/time_distributed/head2d/Head/BN/moving_variance/Initializer/ones*
dtype0
Љ
IModel/time_distributed/head2d/Head/BN/moving_variance/Read/ReadVariableOpReadVariableOp5Model/time_distributed/head2d/Head/BN/moving_variance*
_output_shapes	
:А*
dtype0
Э
4Model/time_distributed/head2d/Head/BN/ReadVariableOpReadVariableOp+Model/time_distributed/head2d/Head/BN/gamma*
_output_shapes	
:А*
dtype0
Ю
6Model/time_distributed/head2d/Head/BN/ReadVariableOp_1ReadVariableOp*Model/time_distributed/head2d/Head/BN/beta*
_output_shapes	
:А*
dtype0
і
EModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOpReadVariableOp1Model/time_distributed/head2d/Head/BN/moving_mean*
_output_shapes	
:А*
dtype0
Ї
GModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5Model/time_distributed/head2d/Head/BN/moving_variance*
_output_shapes	
:А*
dtype0
И
6Model/time_distributed/head2d/Head/BN/FusedBatchNormV3FusedBatchNormV35Model/time_distributed/head2d/Head/Conv2D_1x1/BiasAdd4Model/time_distributed/head2d/Head/BN/ReadVariableOp6Model/time_distributed/head2d/Head/BN/ReadVariableOp_1EModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOpGModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOp_1*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 
І
,Model/time_distributed/head2d/Head/Relu/ReluRelu6Model/time_distributed/head2d/Head/BN/FusedBatchNormV3*
T0*0
_output_shapes
:€€€€€€€€€А
У
BModel/time_distributed/head2d/Head/MaxPool2D/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
№
0Model/time_distributed/head2d/Head/MaxPool2D/MaxMax,Model/time_distributed/head2d/Head/Relu/ReluBModel/time_distributed/head2d/Head/MaxPool2D/Max/reduction_indices*
T0*(
_output_shapes
:€€€€€€€€€А
§
3Model/time_distributed/head2d/Head/Dropout/IdentityIdentity0Model/time_distributed/head2d/Head/MaxPool2D/Max*
T0*(
_output_shapes
:€€€€€€€€€А
я
MModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:*
dtype0*
valueB"      
—
KModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: *
dtype0*
valueB
 *IvЊ
—
KModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: *
dtype0*
valueB
 *Iv>
≠
UModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/RandomUniformRandomUniformMModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/shape*
T0*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:	А*
dtype0
ќ
KModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/subSubKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/maxKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: 
б
KModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/mulMulUModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/RandomUniformKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:	А
”
GModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniformAddKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/mulKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:	А
ц
,Model/time_distributed/head2d/Head/FC/kernelVarHandleOp*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: *
dtype0*
shape:	А*=
shared_name.,Model/time_distributed/head2d/Head/FC/kernel
©
MModel/time_distributed/head2d/Head/FC/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp,Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: 
Ћ
3Model/time_distributed/head2d/Head/FC/kernel/AssignAssignVariableOp,Model/time_distributed/head2d/Head/FC/kernelGModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform*
dtype0
Ѓ
@Model/time_distributed/head2d/Head/FC/kernel/Read/ReadVariableOpReadVariableOp,Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:	А*
dtype0
»
<Model/time_distributed/head2d/Head/FC/bias/Initializer/zerosConst*=
_class3
1/loc:@Model/time_distributed/head2d/Head/FC/bias*
_output_shapes
:*
dtype0*
valueB*    
л
*Model/time_distributed/head2d/Head/FC/biasVarHandleOp*=
_class3
1/loc:@Model/time_distributed/head2d/Head/FC/bias*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Model/time_distributed/head2d/Head/FC/bias
•
KModel/time_distributed/head2d/Head/FC/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp*Model/time_distributed/head2d/Head/FC/bias*
_output_shapes
: 
Љ
1Model/time_distributed/head2d/Head/FC/bias/AssignAssignVariableOp*Model/time_distributed/head2d/Head/FC/bias<Model/time_distributed/head2d/Head/FC/bias/Initializer/zeros*
dtype0
•
>Model/time_distributed/head2d/Head/FC/bias/Read/ReadVariableOpReadVariableOp*Model/time_distributed/head2d/Head/FC/bias*
_output_shapes
:*
dtype0
©
;Model/time_distributed/head2d/Head/FC/MatMul/ReadVariableOpReadVariableOp,Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:	А*
dtype0
Џ
,Model/time_distributed/head2d/Head/FC/MatMulMatMul3Model/time_distributed/head2d/Head/Dropout/Identity;Model/time_distributed/head2d/Head/FC/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
£
<Model/time_distributed/head2d/Head/FC/BiasAdd/ReadVariableOpReadVariableOp*Model/time_distributed/head2d/Head/FC/bias*
_output_shapes
:*
dtype0
÷
-Model/time_distributed/head2d/Head/FC/BiasAddBiasAdd,Model/time_distributed/head2d/Head/FC/MatMul<Model/time_distributed/head2d/Head/FC/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
s
(Model/time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
j
(Model/time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
÷
&Model/time_distributed/Reshape_1/shapePack(Model/time_distributed/Reshape_1/shape/0$Model/time_distributed/strided_slice(Model/time_distributed/Reshape_1/shape/2*
N*
T0*
_output_shapes
:
Ѕ
 Model/time_distributed/Reshape_1Reshape-Model/time_distributed/head2d/Head/FC/BiasAdd&Model/time_distributed/Reshape_1/shape*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

&Model/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      »   
Ч
 Model/time_distributed/Reshape_2Reshapefeature&Model/time_distributed/Reshape_2/shape*
T0*0
_output_shapes
:€€€€€€€€€»
^
Model/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
Д

Model/MeanMean Model/time_distributed/Reshape_1Model/Mean/reduction_indices*
T0*'
_output_shapes
:€€€€€€€€€
W
Model/outputsIdentity
Model/Mean*
T0*'
_output_shapes
:€€€€€€€€€
b
Model/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        
a
Model/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
З
Model/SliceSliceModel/outputsModel/Slice/beginModel/Slice/size*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
T
	Model/mosIdentityModel/Slice*
T0*'
_output_shapes
:€€€€€€€€€

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
f
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Е
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*©
valueЯBЬ	B*Model/time_distributed/head2d/Head/BN/betaB+Model/time_distributed/head2d/Head/BN/gammaB1Model/time_distributed/head2d/Head/BN/moving_meanB5Model/time_distributed/head2d/Head/BN/moving_varianceB2Model/time_distributed/head2d/Head/Conv2D_1x1/biasB4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelB*Model/time_distributed/head2d/Head/FC/biasB,Model/time_distributed/head2d/Head/FC/kernelBglobal_step
Д
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 
”
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices>Model/time_distributed/head2d/Head/BN/beta/Read/ReadVariableOp?Model/time_distributed/head2d/Head/BN/gamma/Read/ReadVariableOpEModel/time_distributed/head2d/Head/BN/moving_mean/Read/ReadVariableOpIModel/time_distributed/head2d/Head/BN/moving_variance/Read/ReadVariableOpFModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Read/ReadVariableOpHModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Read/ReadVariableOp>Model/time_distributed/head2d/Head/FC/bias/Read/ReadVariableOp@Model/time_distributed/head2d/Head/FC/kernel/Read/ReadVariableOpglobal_step/Read/ReadVariableOp"/device:CPU:0*
dtypes
2		
†
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
†
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
И
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*©
valueЯBЬ	B*Model/time_distributed/head2d/Head/BN/betaB+Model/time_distributed/head2d/Head/BN/gammaB1Model/time_distributed/head2d/Head/BN/moving_meanB5Model/time_distributed/head2d/Head/BN/moving_varianceB2Model/time_distributed/head2d/Head/Conv2D_1x1/biasB4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelB*Model/time_distributed/head2d/Head/FC/biasB,Model/time_distributed/head2d/Head/FC/kernelBglobal_step
З
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 
«
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
s
save/AssignVariableOpAssignVariableOp*Model/time_distributed/head2d/Head/BN/betasave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
v
save/AssignVariableOp_1AssignVariableOp+Model/time_distributed/head2d/Head/BN/gammasave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
|
save/AssignVariableOp_2AssignVariableOp1Model/time_distributed/head2d/Head/BN/moving_meansave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
А
save/AssignVariableOp_3AssignVariableOp5Model/time_distributed/head2d/Head/BN/moving_variancesave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
}
save/AssignVariableOp_4AssignVariableOp2Model/time_distributed/head2d/Head/Conv2D_1x1/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:

save/AssignVariableOp_5AssignVariableOp4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
u
save/AssignVariableOp_6AssignVariableOp*Model/time_distributed/head2d/Head/FC/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
w
save/AssignVariableOp_7AssignVariableOp,Model/time_distributed/head2d/Head/FC/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0	*
_output_shapes
:
V
save/AssignVariableOp_8AssignVariableOpglobal_stepsave/Identity_9*
dtype0	
В
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8
-
save/restore_allNoOp^save/restore_shard"√<
save/Const:0save/Identity:0save/restore_all (5 @F8"|
global_stepmk
i
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0"%
saved_model_main_op


group_deps"ь
trainable_variablesдб
Ш
6Model/time_distributed/head2d/Head/Conv2D_1x1/kernel:0;Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/AssignJModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Read/ReadVariableOp:0(2QModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform:08
З
4Model/time_distributed/head2d/Head/Conv2D_1x1/bias:09Model/time_distributed/head2d/Head/Conv2D_1x1/bias/AssignHModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Read/ReadVariableOp:0(2FModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Initializer/zeros:08
к
-Model/time_distributed/head2d/Head/BN/gamma:02Model/time_distributed/head2d/Head/BN/gamma/AssignAModel/time_distributed/head2d/Head/BN/gamma/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/BN/gamma/Initializer/ones:08
з
,Model/time_distributed/head2d/Head/BN/beta:01Model/time_distributed/head2d/Head/BN/beta/Assign@Model/time_distributed/head2d/Head/BN/beta/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/BN/beta/Initializer/zeros:08
ш
.Model/time_distributed/head2d/Head/FC/kernel:03Model/time_distributed/head2d/Head/FC/kernel/AssignBModel/time_distributed/head2d/Head/FC/kernel/Read/ReadVariableOp:0(2IModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform:08
з
,Model/time_distributed/head2d/Head/FC/bias:01Model/time_distributed/head2d/Head/FC/bias/Assign@Model/time_distributed/head2d/Head/FC/bias/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/FC/bias/Initializer/zeros:08"ь
	variablesол
i
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0
Ш
6Model/time_distributed/head2d/Head/Conv2D_1x1/kernel:0;Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/AssignJModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Read/ReadVariableOp:0(2QModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform:08
З
4Model/time_distributed/head2d/Head/Conv2D_1x1/bias:09Model/time_distributed/head2d/Head/Conv2D_1x1/bias/AssignHModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Read/ReadVariableOp:0(2FModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Initializer/zeros:08
к
-Model/time_distributed/head2d/Head/BN/gamma:02Model/time_distributed/head2d/Head/BN/gamma/AssignAModel/time_distributed/head2d/Head/BN/gamma/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/BN/gamma/Initializer/ones:08
з
,Model/time_distributed/head2d/Head/BN/beta:01Model/time_distributed/head2d/Head/BN/beta/Assign@Model/time_distributed/head2d/Head/BN/beta/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/BN/beta/Initializer/zeros:08
Е
3Model/time_distributed/head2d/Head/BN/moving_mean:08Model/time_distributed/head2d/Head/BN/moving_mean/AssignGModel/time_distributed/head2d/Head/BN/moving_mean/Read/ReadVariableOp:0(2EModel/time_distributed/head2d/Head/BN/moving_mean/Initializer/zeros:0@H
Ф
7Model/time_distributed/head2d/Head/BN/moving_variance:0<Model/time_distributed/head2d/Head/BN/moving_variance/AssignKModel/time_distributed/head2d/Head/BN/moving_variance/Read/ReadVariableOp:0(2HModel/time_distributed/head2d/Head/BN/moving_variance/Initializer/ones:0@H
ш
.Model/time_distributed/head2d/Head/FC/kernel:03Model/time_distributed/head2d/Head/FC/kernel/AssignBModel/time_distributed/head2d/Head/FC/kernel/Read/ReadVariableOp:0(2IModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform:08
з
,Model/time_distributed/head2d/Head/FC/bias:01Model/time_distributed/head2d/Head/FC/bias/Assign@Model/time_distributed/head2d/Head/FC/bias/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/FC/bias/Initializer/zeros:08*’
serving_defaultЅ
P
feature_compression9
Placeholder:0&€€€€€€€€€€€€€€€€€€d
N
feature_content;
Placeholder_2:0&€€€€€€€€€€€€€€€€€€d
Q
feature_distortion;
Placeholder_1:0&€€€€€€€€€€€€€€€€€€d.
pred_mos"
Model/mos:0€€€€€€€€€tensorflow/serving/predictЈ®
њЮ
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
,
TPUOrdinalSelector
device_ordinalsИ
Ш
TPUPartitionedCall
args2Tin
device_ordinal
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
autotuner_threshint 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
9
VarIsInitializedOp
resource
is_initialized
И"serve"tpu*2.5.02unknown8Њш

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
К
global_stepVarHandleOp*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	
Ш
PlaceholderPlaceholder*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d*
dtype0*1
shape(:&€€€€€€€€€€€€€€€€€€d
Ъ
Placeholder_1Placeholder*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d*
dtype0*1
shape(:&€€€€€€€€€€€€€€€€€€d
Ъ
Placeholder_2Placeholder*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d*
dtype0*1
shape(:&€€€€€€€€€€€€€€€€€€d
ч
UModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/shapeConst*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
:*
dtype0*%
valueB"      »      
б
SModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/minConst*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: *
dtype0*
valueB
 *хлкљ
б
SModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/maxConst*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: *
dtype0*
valueB
 *хлк=
ќ
]Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/RandomUniformRandomUniformUModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/shape*
T0*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*(
_output_shapes
:»А*
dtype0
о
SModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/subSubSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/maxSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/min*
T0*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: 
К
SModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/mulMul]Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/RandomUniformSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/sub*
T0*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*(
_output_shapes
:»А
ь
OModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniformAddSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/mulSModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform/min*
T0*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*(
_output_shapes
:»А
Ч
4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelVarHandleOp*G
_class=
;9loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: *
dtype0*
shape:»А*E
shared_name64Model/time_distributed/head2d/Head/Conv2D_1x1/kernel
є
UModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp4Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*
_output_shapes
: 
г
;Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/AssignAssignVariableOp4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelOModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform*
dtype0
«
HModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Read/ReadVariableOpReadVariableOp4Model/time_distributed/head2d/Head/Conv2D_1x1/kernel*(
_output_shapes
:»А*
dtype0
Џ
DModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Initializer/zerosConst*E
_class;
97loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/bias*
_output_shapes	
:А*
dtype0*
valueBА*    
Д
2Model/time_distributed/head2d/Head/Conv2D_1x1/biasVarHandleOp*E
_class;
97loc:@Model/time_distributed/head2d/Head/Conv2D_1x1/bias*
_output_shapes
: *
dtype0*
shape:А*C
shared_name42Model/time_distributed/head2d/Head/Conv2D_1x1/bias
µ
SModel/time_distributed/head2d/Head/Conv2D_1x1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp2Model/time_distributed/head2d/Head/Conv2D_1x1/bias*
_output_shapes
: 
‘
9Model/time_distributed/head2d/Head/Conv2D_1x1/bias/AssignAssignVariableOp2Model/time_distributed/head2d/Head/Conv2D_1x1/biasDModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Initializer/zeros*
dtype0
ґ
FModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Read/ReadVariableOpReadVariableOp2Model/time_distributed/head2d/Head/Conv2D_1x1/bias*
_output_shapes	
:А*
dtype0
Ћ
<Model/time_distributed/head2d/Head/BN/gamma/Initializer/onesConst*>
_class4
20loc:@Model/time_distributed/head2d/Head/BN/gamma*
_output_shapes	
:А*
dtype0*
valueBА*  А?
п
+Model/time_distributed/head2d/Head/BN/gammaVarHandleOp*>
_class4
20loc:@Model/time_distributed/head2d/Head/BN/gamma*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Model/time_distributed/head2d/Head/BN/gamma
І
LModel/time_distributed/head2d/Head/BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp+Model/time_distributed/head2d/Head/BN/gamma*
_output_shapes
: 
Њ
2Model/time_distributed/head2d/Head/BN/gamma/AssignAssignVariableOp+Model/time_distributed/head2d/Head/BN/gamma<Model/time_distributed/head2d/Head/BN/gamma/Initializer/ones*
dtype0
®
?Model/time_distributed/head2d/Head/BN/gamma/Read/ReadVariableOpReadVariableOp+Model/time_distributed/head2d/Head/BN/gamma*
_output_shapes	
:А*
dtype0
 
<Model/time_distributed/head2d/Head/BN/beta/Initializer/zerosConst*=
_class3
1/loc:@Model/time_distributed/head2d/Head/BN/beta*
_output_shapes	
:А*
dtype0*
valueBА*    
м
*Model/time_distributed/head2d/Head/BN/betaVarHandleOp*=
_class3
1/loc:@Model/time_distributed/head2d/Head/BN/beta*
_output_shapes
: *
dtype0*
shape:А*;
shared_name,*Model/time_distributed/head2d/Head/BN/beta
•
KModel/time_distributed/head2d/Head/BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp*Model/time_distributed/head2d/Head/BN/beta*
_output_shapes
: 
Љ
1Model/time_distributed/head2d/Head/BN/beta/AssignAssignVariableOp*Model/time_distributed/head2d/Head/BN/beta<Model/time_distributed/head2d/Head/BN/beta/Initializer/zeros*
dtype0
¶
>Model/time_distributed/head2d/Head/BN/beta/Read/ReadVariableOpReadVariableOp*Model/time_distributed/head2d/Head/BN/beta*
_output_shapes	
:А*
dtype0
Ў
CModel/time_distributed/head2d/Head/BN/moving_mean/Initializer/zerosConst*D
_class:
86loc:@Model/time_distributed/head2d/Head/BN/moving_mean*
_output_shapes	
:А*
dtype0*
valueBА*    
Б
1Model/time_distributed/head2d/Head/BN/moving_meanVarHandleOp*D
_class:
86loc:@Model/time_distributed/head2d/Head/BN/moving_mean*
_output_shapes
: *
dtype0*
shape:А*B
shared_name31Model/time_distributed/head2d/Head/BN/moving_mean
≥
RModel/time_distributed/head2d/Head/BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp1Model/time_distributed/head2d/Head/BN/moving_mean*
_output_shapes
: 
—
8Model/time_distributed/head2d/Head/BN/moving_mean/AssignAssignVariableOp1Model/time_distributed/head2d/Head/BN/moving_meanCModel/time_distributed/head2d/Head/BN/moving_mean/Initializer/zeros*
dtype0
і
EModel/time_distributed/head2d/Head/BN/moving_mean/Read/ReadVariableOpReadVariableOp1Model/time_distributed/head2d/Head/BN/moving_mean*
_output_shapes	
:А*
dtype0
я
FModel/time_distributed/head2d/Head/BN/moving_variance/Initializer/onesConst*H
_class>
<:loc:@Model/time_distributed/head2d/Head/BN/moving_variance*
_output_shapes	
:А*
dtype0*
valueBА*  А?
Н
5Model/time_distributed/head2d/Head/BN/moving_varianceVarHandleOp*H
_class>
<:loc:@Model/time_distributed/head2d/Head/BN/moving_variance*
_output_shapes
: *
dtype0*
shape:А*F
shared_name75Model/time_distributed/head2d/Head/BN/moving_variance
ї
VModel/time_distributed/head2d/Head/BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp5Model/time_distributed/head2d/Head/BN/moving_variance*
_output_shapes
: 
№
<Model/time_distributed/head2d/Head/BN/moving_variance/AssignAssignVariableOp5Model/time_distributed/head2d/Head/BN/moving_varianceFModel/time_distributed/head2d/Head/BN/moving_variance/Initializer/ones*
dtype0
Љ
IModel/time_distributed/head2d/Head/BN/moving_variance/Read/ReadVariableOpReadVariableOp5Model/time_distributed/head2d/Head/BN/moving_variance*
_output_shapes	
:А*
dtype0
я
MModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:*
dtype0*
valueB"      
—
KModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: *
dtype0*
valueB
 *IvЊ
—
KModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: *
dtype0*
valueB
 *Iv>
≠
UModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/RandomUniformRandomUniformMModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/shape*
T0*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:	А*
dtype0
ќ
KModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/subSubKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/maxKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: 
б
KModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/mulMulUModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/RandomUniformKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:	А
”
GModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniformAddKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/mulKModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:	А
ц
,Model/time_distributed/head2d/Head/FC/kernelVarHandleOp*?
_class5
31loc:@Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: *
dtype0*
shape:	А*=
shared_name.,Model/time_distributed/head2d/Head/FC/kernel
©
MModel/time_distributed/head2d/Head/FC/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp,Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
: 
Ћ
3Model/time_distributed/head2d/Head/FC/kernel/AssignAssignVariableOp,Model/time_distributed/head2d/Head/FC/kernelGModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform*
dtype0
Ѓ
@Model/time_distributed/head2d/Head/FC/kernel/Read/ReadVariableOpReadVariableOp,Model/time_distributed/head2d/Head/FC/kernel*
_output_shapes
:	А*
dtype0
»
<Model/time_distributed/head2d/Head/FC/bias/Initializer/zerosConst*=
_class3
1/loc:@Model/time_distributed/head2d/Head/FC/bias*
_output_shapes
:*
dtype0*
valueB*    
л
*Model/time_distributed/head2d/Head/FC/biasVarHandleOp*=
_class3
1/loc:@Model/time_distributed/head2d/Head/FC/bias*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Model/time_distributed/head2d/Head/FC/bias
•
KModel/time_distributed/head2d/Head/FC/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp*Model/time_distributed/head2d/Head/FC/bias*
_output_shapes
: 
Љ
1Model/time_distributed/head2d/Head/FC/bias/AssignAssignVariableOp*Model/time_distributed/head2d/Head/FC/bias<Model/time_distributed/head2d/Head/FC/bias/Initializer/zeros*
dtype0
•
>Model/time_distributed/head2d/Head/FC/bias/Read/ReadVariableOpReadVariableOp*Model/time_distributed/head2d/Head/FC/bias*
_output_shapes
:*
dtype0
M
TPUOrdinalSelectorTPUOrdinalSelector*#
_output_shapes
:€€€€€€€€€
„
TPUPartitionedCallTPUPartitionedCallPlaceholderPlaceholder_1Placeholder_24Model/time_distributed/head2d/Head/Conv2D_1x1/kernel2Model/time_distributed/head2d/Head/Conv2D_1x1/bias+Model/time_distributed/head2d/Head/BN/gamma*Model/time_distributed/head2d/Head/BN/beta1Model/time_distributed/head2d/Head/BN/moving_mean5Model/time_distributed/head2d/Head/BN/moving_variance,Model/time_distributed/head2d/Head/FC/kernel*Model/time_distributed/head2d/Head/FC/biasTPUOrdinalSelector*
Tin
2*
Tout
2*
_output_shapes

::*!
fR
tpu_subgraph_2a5mCaRAQ9E

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
f
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Е
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*©
valueЯBЬ	B*Model/time_distributed/head2d/Head/BN/betaB+Model/time_distributed/head2d/Head/BN/gammaB1Model/time_distributed/head2d/Head/BN/moving_meanB5Model/time_distributed/head2d/Head/BN/moving_varianceB2Model/time_distributed/head2d/Head/Conv2D_1x1/biasB4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelB*Model/time_distributed/head2d/Head/FC/biasB,Model/time_distributed/head2d/Head/FC/kernelBglobal_step
Д
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 
”
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices>Model/time_distributed/head2d/Head/BN/beta/Read/ReadVariableOp?Model/time_distributed/head2d/Head/BN/gamma/Read/ReadVariableOpEModel/time_distributed/head2d/Head/BN/moving_mean/Read/ReadVariableOpIModel/time_distributed/head2d/Head/BN/moving_variance/Read/ReadVariableOpFModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Read/ReadVariableOpHModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Read/ReadVariableOp>Model/time_distributed/head2d/Head/FC/bias/Read/ReadVariableOp@Model/time_distributed/head2d/Head/FC/kernel/Read/ReadVariableOpglobal_step/Read/ReadVariableOp"/device:CPU:0*
dtypes
2		
†
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
†
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
И
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*©
valueЯBЬ	B*Model/time_distributed/head2d/Head/BN/betaB+Model/time_distributed/head2d/Head/BN/gammaB1Model/time_distributed/head2d/Head/BN/moving_meanB5Model/time_distributed/head2d/Head/BN/moving_varianceB2Model/time_distributed/head2d/Head/Conv2D_1x1/biasB4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelB*Model/time_distributed/head2d/Head/FC/biasB,Model/time_distributed/head2d/Head/FC/kernelBglobal_step
З
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 
«
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
s
save/AssignVariableOpAssignVariableOp*Model/time_distributed/head2d/Head/BN/betasave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
v
save/AssignVariableOp_1AssignVariableOp+Model/time_distributed/head2d/Head/BN/gammasave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
|
save/AssignVariableOp_2AssignVariableOp1Model/time_distributed/head2d/Head/BN/moving_meansave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
А
save/AssignVariableOp_3AssignVariableOp5Model/time_distributed/head2d/Head/BN/moving_variancesave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
}
save/AssignVariableOp_4AssignVariableOp2Model/time_distributed/head2d/Head/Conv2D_1x1/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:

save/AssignVariableOp_5AssignVariableOp4Model/time_distributed/head2d/Head/Conv2D_1x1/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
u
save/AssignVariableOp_6AssignVariableOp*Model/time_distributed/head2d/Head/FC/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
w
save/AssignVariableOp_7AssignVariableOp,Model/time_distributed/head2d/Head/FC/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0	*
_output_shapes
:
V
save/AssignVariableOp_8AssignVariableOpglobal_stepsave/Identity_9*
dtype0	
В
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8
-
save/restore_allNoOp^save/restore_shardГ\
А\
Ј
tpu_subgraph_2a5mCaRAQ9E#
feature_compression_placeholder"
feature_distortion_placeholder
feature_content_placeholderS
Omodel_time_distributed_head2d_head_conv2d_1x1_conv2d_readvariableop_placeholderT
Pmodel_time_distributed_head2d_head_conv2d_1x1_biasadd_readvariableop_placeholderD
@model_time_distributed_head2d_head_bn_readvariableop_placeholderF
Bmodel_time_distributed_head2d_head_bn_readvariableop_1_placeholderU
Qmodel_time_distributed_head2d_head_bn_fusedbatchnormv3_readvariableop_placeholderW
Smodel_time_distributed_head2d_head_bn_fusedbatchnormv3_readvariableop_1_placeholderK
Gmodel_time_distributed_head2d_head_fc_matmul_readvariableop_placeholderL
Hmodel_time_distributed_head2d_head_fc_biasadd_readvariableop_placeholder
output_0_shard_0
output_1_shard_0ИG
cluster/pivotNoOp*
_pivot_for_cluster	cluster2
cluster/pivot™
TPUReplicateMetadataTPUReplicateMetadata^cluster/pivot*
_tpu_replicate	cluster*
num_replicas*#
use_spmd_for_xla_partitioning(2
TPUReplicateMetadataФ
feature_compressionIdentityfeature_compression_placeholder^TPUReplicateMetadata*
T0*
_tpu_replicate	cluster2
feature_compressionС
feature_distortionIdentityfeature_distortion_placeholder^TPUReplicateMetadata*
T0*
_tpu_replicate	cluster2
feature_distortionИ
feature_contentIdentityfeature_content_placeholder^TPUReplicateMetadata*
T0*
_tpu_replicate	cluster2
feature_contentx
concat/axisConst^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
value	B :2
concat/axisЩ
concatConcatV2feature_compression:output:0feature_content:output:0concat/axis:output:0*
N*
T0*
_tpu_replicate	cluster2
concatU
featureIdentityconcat:output:0*
T0*
_tpu_replicate	cluster2	
feature}
Model/time_distributed/ShapeShapefeature:output:0*
T0*
_tpu_replicate	cluster2
Model/time_distributed/ShapeЇ
*Model/time_distributed/strided_slice/stackConst^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
valueB:2,
*Model/time_distributed/strided_slice/stackЊ
,Model/time_distributed/strided_slice/stack_1Const^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
valueB:2.
,Model/time_distributed/strided_slice/stack_1Њ
,Model/time_distributed/strided_slice/stack_2Const^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
valueB:2.
,Model/time_distributed/strided_slice/stack_2с
$Model/time_distributed/strided_sliceStridedSlice%Model/time_distributed/Shape:output:03Model/time_distributed/strided_slice/stack:output:05Model/time_distributed/strided_slice/stack_1:output:05Model/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_tpu_replicate	cluster*
shrink_axis_mask2&
$Model/time_distributed/strided_sliceљ
$Model/time_distributed/Reshape/shapeConst^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*%
valueB"€€€€      »   2&
$Model/time_distributed/Reshape/shape≤
Model/time_distributed/ReshapeReshapefeature:output:0-Model/time_distributed/Reshape/shape:output:0*
T0*
_tpu_replicate	cluster2 
Model/time_distributed/ReshapeЃ
CModel/time_distributed/head2d/Head/Conv2D_1x1/Conv2D/ReadVariableOpReadVariableOpOmodel_time_distributed_head2d_head_conv2d_1x1_conv2d_readvariableop_placeholder^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype02E
CModel/time_distributed/head2d/Head/Conv2D_1x1/Conv2D/ReadVariableOpЇ
4Model/time_distributed/head2d/Head/Conv2D_1x1/Conv2DConv2D'Model/time_distributed/Reshape:output:0KModel/time_distributed/head2d/Head/Conv2D_1x1/Conv2D/ReadVariableOp:value:0*
T0*
_tpu_replicate	cluster*
paddingSAME*
strides
26
4Model/time_distributed/head2d/Head/Conv2D_1x1/Conv2D±
DModel/time_distributed/head2d/Head/Conv2D_1x1/BiasAdd/ReadVariableOpReadVariableOpPmodel_time_distributed_head2d_head_conv2d_1x1_biasadd_readvariableop_placeholder^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype02F
DModel/time_distributed/head2d/Head/Conv2D_1x1/BiasAdd/ReadVariableOpђ
5Model/time_distributed/head2d/Head/Conv2D_1x1/BiasAddBiasAdd=Model/time_distributed/head2d/Head/Conv2D_1x1/Conv2D:output:0LModel/time_distributed/head2d/Head/Conv2D_1x1/BiasAdd/ReadVariableOp:value:0*
T0*
_tpu_replicate	cluster27
5Model/time_distributed/head2d/Head/Conv2D_1x1/BiasAddБ
4Model/time_distributed/head2d/Head/BN/ReadVariableOpReadVariableOp@model_time_distributed_head2d_head_bn_readvariableop_placeholder^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype026
4Model/time_distributed/head2d/Head/BN/ReadVariableOpЗ
6Model/time_distributed/head2d/Head/BN/ReadVariableOp_1ReadVariableOpBmodel_time_distributed_head2d_head_bn_readvariableop_1_placeholder^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype028
6Model/time_distributed/head2d/Head/BN/ReadVariableOp_1і
EModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOpReadVariableOpQmodel_time_distributed_head2d_head_bn_fusedbatchnormv3_readvariableop_placeholder^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype02G
EModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOpЇ
GModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpSmodel_time_distributed_head2d_head_bn_fusedbatchnormv3_readvariableop_1_placeholder^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype02I
GModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOp_1ґ
6Model/time_distributed/head2d/Head/BN/FusedBatchNormV3FusedBatchNormV3>Model/time_distributed/head2d/Head/Conv2D_1x1/BiasAdd:output:0<Model/time_distributed/head2d/Head/BN/ReadVariableOp:value:0>Model/time_distributed/head2d/Head/BN/ReadVariableOp_1:value:0MModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOp:value:0OModel/time_distributed/head2d/Head/BN/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
_tpu_replicate	cluster*
epsilon%oГ:*
is_training( 28
6Model/time_distributed/head2d/Head/BN/FusedBatchNormV3∆
,Model/time_distributed/head2d/Head/Relu/ReluRelu:Model/time_distributed/head2d/Head/BN/FusedBatchNormV3:y:0*
T0*
_tpu_replicate	cluster2.
,Model/time_distributed/head2d/Head/Relu/Reluс
BModel/time_distributed/head2d/Head/MaxPool2D/Max/reduction_indicesConst^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
valueB"      2D
BModel/time_distributed/head2d/Head/MaxPool2D/Max/reduction_indicesЪ
0Model/time_distributed/head2d/Head/MaxPool2D/MaxMax:Model/time_distributed/head2d/Head/Relu/Relu:activations:0KModel/time_distributed/head2d/Head/MaxPool2D/Max/reduction_indices:output:0*
T0*
_tpu_replicate	cluster22
0Model/time_distributed/head2d/Head/MaxPool2D/Max„
3Model/time_distributed/head2d/Head/Dropout/IdentityIdentity9Model/time_distributed/head2d/Head/MaxPool2D/Max:output:0*
T0*
_tpu_replicate	cluster25
3Model/time_distributed/head2d/Head/Dropout/IdentityЦ
;Model/time_distributed/head2d/Head/FC/MatMul/ReadVariableOpReadVariableOpGmodel_time_distributed_head2d_head_fc_matmul_readvariableop_placeholder^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype02=
;Model/time_distributed/head2d/Head/FC/MatMul/ReadVariableOpП
,Model/time_distributed/head2d/Head/FC/MatMulMatMul<Model/time_distributed/head2d/Head/Dropout/Identity:output:0CModel/time_distributed/head2d/Head/FC/MatMul/ReadVariableOp:value:0*
T0*
_tpu_replicate	cluster2.
,Model/time_distributed/head2d/Head/FC/MatMulЩ
<Model/time_distributed/head2d/Head/FC/BiasAdd/ReadVariableOpReadVariableOpHmodel_time_distributed_head2d_head_fc_biasadd_readvariableop_placeholder^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype02>
<Model/time_distributed/head2d/Head/FC/BiasAdd/ReadVariableOpН
-Model/time_distributed/head2d/Head/FC/BiasAddBiasAdd6Model/time_distributed/head2d/Head/FC/MatMul:product:0DModel/time_distributed/head2d/Head/FC/BiasAdd/ReadVariableOp:value:0*
T0*
_tpu_replicate	cluster2/
-Model/time_distributed/head2d/Head/FC/BiasAddї
(Model/time_distributed/Reshape_1/shape/0Const^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
valueB :
€€€€€€€€€2*
(Model/time_distributed/Reshape_1/shape/0≤
(Model/time_distributed/Reshape_1/shape/2Const^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
value	B :2*
(Model/time_distributed/Reshape_1/shape/2Ь
&Model/time_distributed/Reshape_1/shapePack1Model/time_distributed/Reshape_1/shape/0:output:0-Model/time_distributed/strided_slice:output:01Model/time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_tpu_replicate	cluster2(
&Model/time_distributed/Reshape_1/shapeё
 Model/time_distributed/Reshape_1Reshape6Model/time_distributed/head2d/Head/FC/BiasAdd:output:0/Model/time_distributed/Reshape_1/shape:output:0*
T0*
_tpu_replicate	cluster2"
 Model/time_distributed/Reshape_1Ѕ
&Model/time_distributed/Reshape_2/shapeConst^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*%
valueB"€€€€      »   2(
&Model/time_distributed/Reshape_2/shapeЄ
 Model/time_distributed/Reshape_2Reshapefeature:output:0/Model/time_distributed/Reshape_2/shape:output:0*
T0*
_tpu_replicate	cluster2"
 Model/time_distributed/Reshape_2Ъ
Model/Mean/reduction_indicesConst^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
value	B :2
Model/Mean/reduction_indicesШ

Model/MeanMean)Model/time_distributed/Reshape_1:output:0%Model/Mean/reduction_indices:output:0*
T0*
_tpu_replicate	cluster2

Model/Meane
Model/outputsIdentityModel/Mean:output:0*
T0*
_tpu_replicate	cluster2
Model/outputsП
Model/Slice/beginConst^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
valueB"        2
Model/Slice/beginН
Model/Slice/sizeConst^TPUReplicateMetadata*
_tpu_replicate	cluster*
dtype0*
valueB"€€€€   2
Model/Slice/size•
Model/SliceSliceModel/outputs:output:0Model/Slice/begin:output:0Model/Slice/size:output:0*
Index0*
T0*
_tpu_replicate	cluster2
Model/Slice^
	Model/mosIdentityModel/Slice:output:0*
T0*
_tpu_replicate	cluster2
	Model/mosA
NoOpNoOp^cluster/pivot*
_tpu_replicate	cluster2
NoOpХ
IdentityIdentityModel/mos:output:0"/device:TPU_REPLICATED_CORE:0*
T0*
_tpu_output_identity(*
_tpu_replicate	cluster2

IdentityЩ

Identity_1IdentityModel/mos:output:0"/device:TPU_REPLICATED_CORE:0*
T0*
_tpu_output_identity(*
_tpu_replicate	cluster2

Identity_1Б
TPUCompilationResultTPUCompilationResult^TPUReplicateMetadata*$
_tpu_compilation_status	cluster2
TPUCompilationResultY
output0TPUReplicatedOutputIdentity:output:0*
T0*
num_replicas2	
output0U
output_0_shard_0_0Identityoutput0:outputs:0^NoOp*
T02
output_0_shard_0[
output1TPUReplicatedOutputIdentity_1:output:0*
T0*
num_replicas2	
output1U
output_1_shard_0_0Identityoutput1:outputs:0^NoOp*
T02
output_1_shard_0"/
output_0_shard_0output_0_shard_0_0:output:0"/
output_1_shard_0output_1_shard_0_0:output:0*#
_disable_call_shape_inference(:B >
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d:B>
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d:B>
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€d"√<
save/Const:0save/Identity:0save/restore_all (5 @F8"|
global_stepmk
i
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0"%
saved_model_main_op


group_deps"ь
trainable_variablesдб
Ш
6Model/time_distributed/head2d/Head/Conv2D_1x1/kernel:0;Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/AssignJModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Read/ReadVariableOp:0(2QModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform:08
З
4Model/time_distributed/head2d/Head/Conv2D_1x1/bias:09Model/time_distributed/head2d/Head/Conv2D_1x1/bias/AssignHModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Read/ReadVariableOp:0(2FModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Initializer/zeros:08
к
-Model/time_distributed/head2d/Head/BN/gamma:02Model/time_distributed/head2d/Head/BN/gamma/AssignAModel/time_distributed/head2d/Head/BN/gamma/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/BN/gamma/Initializer/ones:08
з
,Model/time_distributed/head2d/Head/BN/beta:01Model/time_distributed/head2d/Head/BN/beta/Assign@Model/time_distributed/head2d/Head/BN/beta/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/BN/beta/Initializer/zeros:08
ш
.Model/time_distributed/head2d/Head/FC/kernel:03Model/time_distributed/head2d/Head/FC/kernel/AssignBModel/time_distributed/head2d/Head/FC/kernel/Read/ReadVariableOp:0(2IModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform:08
з
,Model/time_distributed/head2d/Head/FC/bias:01Model/time_distributed/head2d/Head/FC/bias/Assign@Model/time_distributed/head2d/Head/FC/bias/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/FC/bias/Initializer/zeros:08"ь
	variablesол
i
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0
Ш
6Model/time_distributed/head2d/Head/Conv2D_1x1/kernel:0;Model/time_distributed/head2d/Head/Conv2D_1x1/kernel/AssignJModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Read/ReadVariableOp:0(2QModel/time_distributed/head2d/Head/Conv2D_1x1/kernel/Initializer/random_uniform:08
З
4Model/time_distributed/head2d/Head/Conv2D_1x1/bias:09Model/time_distributed/head2d/Head/Conv2D_1x1/bias/AssignHModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Read/ReadVariableOp:0(2FModel/time_distributed/head2d/Head/Conv2D_1x1/bias/Initializer/zeros:08
к
-Model/time_distributed/head2d/Head/BN/gamma:02Model/time_distributed/head2d/Head/BN/gamma/AssignAModel/time_distributed/head2d/Head/BN/gamma/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/BN/gamma/Initializer/ones:08
з
,Model/time_distributed/head2d/Head/BN/beta:01Model/time_distributed/head2d/Head/BN/beta/Assign@Model/time_distributed/head2d/Head/BN/beta/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/BN/beta/Initializer/zeros:08
Е
3Model/time_distributed/head2d/Head/BN/moving_mean:08Model/time_distributed/head2d/Head/BN/moving_mean/AssignGModel/time_distributed/head2d/Head/BN/moving_mean/Read/ReadVariableOp:0(2EModel/time_distributed/head2d/Head/BN/moving_mean/Initializer/zeros:0@H
Ф
7Model/time_distributed/head2d/Head/BN/moving_variance:0<Model/time_distributed/head2d/Head/BN/moving_variance/AssignKModel/time_distributed/head2d/Head/BN/moving_variance/Read/ReadVariableOp:0(2HModel/time_distributed/head2d/Head/BN/moving_variance/Initializer/ones:0@H
ш
.Model/time_distributed/head2d/Head/FC/kernel:03Model/time_distributed/head2d/Head/FC/kernel/AssignBModel/time_distributed/head2d/Head/FC/kernel/Read/ReadVariableOp:0(2IModel/time_distributed/head2d/Head/FC/kernel/Initializer/random_uniform:08
з
,Model/time_distributed/head2d/Head/FC/bias:01Model/time_distributed/head2d/Head/FC/bias/Assign@Model/time_distributed/head2d/Head/FC/bias/Read/ReadVariableOp:0(2>Model/time_distributed/head2d/Head/FC/bias/Initializer/zeros:08*ѕ
serving_defaultї
P
feature_compression9
Placeholder:0&€€€€€€€€€€€€€€€€€€d
N
feature_content;
Placeholder_2:0&€€€€€€€€€€€€€€€€€€d
Q
feature_distortion;
Placeholder_1:0&€€€€€€€€€€€€€€€€€€d(
pred_mos
TPUPartitionedCall:1tensorflow/serving/predict