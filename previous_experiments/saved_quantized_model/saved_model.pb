��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.12v2.14.0-10-g99d80a9e2548��
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:
*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*&
shared_nameAdam/dense_1/kernel/v
�
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	�
*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
��*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:
*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*&
shared_nameAdam/dense_1/kernel/m
�
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	�
*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
��*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�
*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
�
 quant_dense_1/pre_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_dense_1/pre_activation_max
�
4quant_dense_1/pre_activation_max/Read/ReadVariableOpReadVariableOp quant_dense_1/pre_activation_max*
_output_shapes
: *
dtype0
�
 quant_dense_1/pre_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_dense_1/pre_activation_min
�
4quant_dense_1/pre_activation_min/Read/ReadVariableOpReadVariableOp quant_dense_1/pre_activation_min*
_output_shapes
: *
dtype0
�
quant_dense_1/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_1/kernel_max
}
,quant_dense_1/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_1/kernel_max*
_output_shapes
: *
dtype0
�
quant_dense_1/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_1/kernel_min
}
,quant_dense_1/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_1/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_1/optimizer_step
�
0quant_dense_1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_1/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!quant_dense/post_activation_max
�
3quant_dense/post_activation_max/Read/ReadVariableOpReadVariableOpquant_dense/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!quant_dense/post_activation_min
�
3quant_dense/post_activation_min/Read/ReadVariableOpReadVariableOpquant_dense/post_activation_min*
_output_shapes
: *
dtype0
�
quant_dense/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_dense/kernel_max
y
*quant_dense/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense/kernel_max*
_output_shapes
: *
dtype0
�
quant_dense/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_dense/kernel_min
y
*quant_dense/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namequant_dense/optimizer_step
�
.quant_dense/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense/optimizer_step*
_output_shapes
: *
dtype0
�
quant_flatten/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_flatten/optimizer_step
�
0quant_flatten/optimizer_step/Read/ReadVariableOpReadVariableOpquant_flatten/optimizer_step*
_output_shapes
: *
dtype0
�
quantize_layer/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequantize_layer/optimizer_step
�
1quantize_layer/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer/optimizer_step*
_output_shapes
: *
dtype0
�
!quantize_layer/quantize_layer_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_max
�
5quantize_layer/quantize_layer_max/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_max*
_output_shapes
: *
dtype0
�
!quantize_layer/quantize_layer_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_min
�
5quantize_layer/quantize_layer_min/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_min*
_output_shapes
: *
dtype0
�
serving_default_flatten_inputPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_input!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxdense/kernelquant_dense/kernel_minquant_dense/kernel_max
dense/biasquant_dense/post_activation_minquant_dense/post_activation_maxdense_1/kernelquant_dense_1/kernel_minquant_dense_1/kernel_maxdense_1/bias quant_dense_1/pre_activation_min quant_dense_1/pre_activation_max*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_181563

NoOpNoOp
�M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�M
value�LB�L B�L
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer
optimizer_step
 _weight_vars
!_quantize_activations
"_output_quantizers*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
	)layer
*optimizer_step
+_weight_vars
,
kernel_min
-
kernel_max
._quantize_activations
/post_activation_min
0post_activation_max
1_output_quantizers*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
	8layer
9optimizer_step
:_weight_vars
;
kernel_min
<
kernel_max
=_quantize_activations
>pre_activation_min
?pre_activation_max
@_output_quantizers*
�
0
1
2
3
A4
B5
*6
,7
-8
/9
010
C11
D12
913
;14
<15
>16
?17*
 
A0
B1
C2
D3*
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*

Jtrace_0
Ktrace_1* 

Ltrace_0
Mtrace_1* 
* 
�
Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_rateAm�Bm�Cm�Dm�Av�Bv�Cv�Dv�*

Sserving_default* 

0
1
2*
* 
* 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ytrace_0
Ztrace_1* 

[trace_0
\trace_1* 
}w
VARIABLE_VALUE!quantize_layer/quantize_layer_minBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!quantize_layer/quantize_layer_maxBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUE*

min_var
max_var*
uo
VARIABLE_VALUEquantize_layer/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

0*
* 
* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

btrace_0
ctrace_1* 

dtrace_0
etrace_1* 
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses* 
tn
VARIABLE_VALUEquant_flatten/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
5
A0
B1
*2
,3
-4
/5
06*

A0
B1*
* 
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

qtrace_0
rtrace_1* 

strace_0
ttrace_1* 
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

Akernel
Bbias*
rl
VARIABLE_VALUEquant_dense/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

{0*
jd
VARIABLE_VALUEquant_dense/kernel_min:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEquant_dense/kernel_max:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
|v
VARIABLE_VALUEquant_dense/post_activation_minClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEquant_dense/post_activation_maxClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
C0
D1
92
;3
<4
>5
?6*

C0
D1*
* 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ckernel
Dbias*
tn
VARIABLE_VALUEquant_dense_1/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

�0*
lf
VARIABLE_VALUEquant_dense_1/kernel_min:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEquant_dense_1/kernel_max:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
|v
VARIABLE_VALUE quant_dense_1/pre_activation_minBlayer_with_weights-3/pre_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE quant_dense_1/pre_activation_maxBlayer_with_weights-3/pre_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
LF
VARIABLE_VALUEdense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
j
0
1
2
3
*4
,5
-6
/7
08
99
;10
<11
>12
?13*
 
0
1
2
3*

�0
�1*
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 

0*
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 
* 
* 
'
*0
,1
-2
/3
04*

)0*
* 
* 
* 
* 
* 
* 
* 

B0*

B0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 

A0
�2*
'
90
;1
<2
>3
?4*

80*
* 
* 
* 
* 
* 
* 
* 

D0*

D0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

C0
�2*
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

,min_var
-max_var*
* 
* 
* 
* 
* 

;min_var
<max_var*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
oi
VARIABLE_VALUEAdam/dense/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/dense/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/dense/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxquantize_layer/optimizer_stepquant_flatten/optimizer_stepquant_dense/optimizer_stepquant_dense/kernel_minquant_dense/kernel_maxquant_dense/post_activation_minquant_dense/post_activation_maxquant_dense_1/optimizer_stepquant_dense_1/kernel_minquant_dense_1/kernel_max quant_dense_1/pre_activation_min quant_dense_1/pre_activation_maxdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vConst*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_182073
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxquantize_layer/optimizer_stepquant_flatten/optimizer_stepquant_dense/optimizer_stepquant_dense/kernel_minquant_dense/kernel_maxquant_dense/post_activation_minquant_dense/post_activation_maxquant_dense_1/optimizer_stepquant_dense_1/kernel_minquant_dense_1/kernel_max quant_dense_1/pre_activation_min quant_dense_1/pre_activation_maxdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_182187�

�
�
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181611

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�Q
�
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181820

inputsB
/lastvaluequant_batchmin_readvariableop_resource:	�
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:
@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
y
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�"
�
G__inference_quant_dense_layer_call_and_return_conditional_losses_181737

inputsR
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
��J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: .
biasadd_readvariableop_resource:	�K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
��*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:�����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_quantize_layer_layer_call_fn_181581

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181279w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181577:&"
 
_user_specified_name181575:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181627

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_181394
flatten_input
unknown: 
	unknown_0: 
	unknown_1:
��
	unknown_2: 
	unknown_3: 
	unknown_4:	�
	unknown_5: 
	unknown_6: 
	unknown_7:	�

	unknown_8: 
	unknown_9: 

unknown_10:


unknown_11: 

unknown_12: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_181268o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181390:&"
 
_user_specified_name181388:&"
 
_user_specified_name181386:&"
 
_user_specified_name181384:&
"
 
_user_specified_name181382:&	"
 
_user_specified_name181380:&"
 
_user_specified_name181378:&"
 
_user_specified_name181376:&"
 
_user_specified_name181374:&"
 
_user_specified_name181372:&"
 
_user_specified_name181370:&"
 
_user_specified_name181368:&"
 
_user_specified_name181366:&"
 
_user_specified_name181364:^ Z
/
_output_shapes
:���������
'
_user_specified_nameflatten_input
�g
�
!__inference__wrapped_model_181107
flatten_inpute
[sequential_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: g
]sequential_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: i
Usequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
��a
Wsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: a
Wsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
6sequential_quant_dense_biasadd_readvariableop_resource:	�b
Xsequential_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: d
Zsequential_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: j
Wsequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	�
c
Ysequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: c
Ysequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: F
8sequential_quant_dense_1_biasadd_readvariableop_resource:
d
Zsequential_quant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: f
\sequential_quant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��-sequential/quant_dense/BiasAdd/ReadVariableOp�Lsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Osequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Qsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�/sequential/quant_dense_1/BiasAdd/ReadVariableOp�Nsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Qsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Ssequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Rsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Tsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Rsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp[sequential_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Tsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp]sequential_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Csequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsflatten_inputZsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0\sequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:���������o
sequential/quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  �
 sequential/quant_flatten/ReshapeReshapeMsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0'sequential/quant_flatten/Const:output:0*
T0*(
_output_shapes
:�����������
Lsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpWsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
=sequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsTsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Vsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Vsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
��*
narrow_range(�
sequential/quant_dense/MatMulMatMul)sequential/quant_flatten/Reshape:output:0Gsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:�����������
-sequential/quant_dense/BiasAdd/ReadVariableOpReadVariableOp6sequential_quant_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/quant_dense/BiasAddBiasAdd'sequential/quant_dense/MatMul:product:05sequential/quant_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential/quant_dense/ReluRelu'sequential/quant_dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Osequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpXsequential_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Qsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpZsequential_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
@sequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars)sequential/quant_dense/Relu:activations:0Wsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Ysequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:�����������
Nsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWsequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYsequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpYsequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
?sequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsVsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Xsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Xsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(�
sequential/quant_dense_1/MatMulMatMulJsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Isequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
�
/sequential/quant_dense_1/BiasAdd/ReadVariableOpReadVariableOp8sequential_quant_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
 sequential/quant_dense_1/BiasAddBiasAdd)sequential/quant_dense_1/MatMul:product:07sequential/quant_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
Qsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpZsequential_quant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Ssequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp\sequential_quant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Bsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars)sequential/quant_dense_1/BiasAdd:output:0Ysequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0[sequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
�
 sequential/quant_dense_1/SoftmaxSoftmaxLsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
y
IdentityIdentity*sequential/quant_dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp.^sequential/quant_dense/BiasAdd/ReadVariableOpM^sequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpO^sequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1O^sequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2P^sequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpR^sequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_10^sequential/quant_dense_1/BiasAdd/ReadVariableOpO^sequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpQ^sequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Q^sequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2R^sequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpT^sequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1S^sequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpU^sequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 2^
-sequential/quant_dense/BiasAdd/ReadVariableOp-sequential/quant_dense/BiasAdd/ReadVariableOp2�
Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Lsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpLsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Qsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Qsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Osequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpOsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2b
/sequential/quant_dense_1/BiasAdd/ReadVariableOp/sequential/quant_dense_1/BiasAdd/ReadVariableOp2�
Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Nsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpNsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Ssequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ssequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Qsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Tsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Tsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Rsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpRsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
/
_output_shapes
:���������
'
_user_specified_nameflatten_input
�Q
�
G__inference_quant_dense_layer_call_and_return_conditional_losses_181191

inputsC
/lastvaluequant_batchmin_readvariableop_resource:
��/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: .
biasadd_readvariableop_resource:	�@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
��*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:�����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Q
�
G__inference_quant_dense_layer_call_and_return_conditional_losses_181716

inputsC
/lastvaluequant_batchmin_readvariableop_resource:
��/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: .
biasadd_readvariableop_resource:	�@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
��*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:�����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
J
.__inference_quant_flatten_layer_call_fn_181621

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181290a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�"
�
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181841

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	�
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:
K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
y
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�"
�
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181346

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	�
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:
K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
y
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181279

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_181427
flatten_input
unknown: 
	unknown_0: 
	unknown_1:
��
	unknown_2: 
	unknown_3: 
	unknown_4:	�
	unknown_5: 
	unknown_6: 
	unknown_7:	�

	unknown_8: 
	unknown_9: 

unknown_10:


unknown_11: 

unknown_12: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_181361o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181423:&"
 
_user_specified_name181421:&"
 
_user_specified_name181419:&"
 
_user_specified_name181417:&
"
 
_user_specified_name181415:&	"
 
_user_specified_name181413:&"
 
_user_specified_name181411:&"
 
_user_specified_name181409:&"
 
_user_specified_name181407:&"
 
_user_specified_name181405:&"
 
_user_specified_name181403:&"
 
_user_specified_name181401:&"
 
_user_specified_name181399:&"
 
_user_specified_name181397:^ Z
/
_output_shapes
:���������
'
_user_specified_nameflatten_input
�
�
F__inference_sequential_layer_call_and_return_conditional_losses_181268
flatten_input
quantize_layer_181131: 
quantize_layer_181133: &
quant_dense_181192:
��
quant_dense_181194: 
quant_dense_181196: !
quant_dense_181198:	�
quant_dense_181200: 
quant_dense_181202: '
quant_dense_1_181254:	�

quant_dense_1_181256: 
quant_dense_1_181258: "
quant_dense_1_181260:

quant_dense_1_181262: 
quant_dense_1_181264: 
identity��#quant_dense/StatefulPartitionedCall�%quant_dense_1/StatefulPartitionedCall�&quantize_layer/StatefulPartitionedCall�
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallflatten_inputquantize_layer_181131quantize_layer_181133*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181130�
quant_flatten/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181141�
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_181192quant_dense_181194quant_dense_181196quant_dense_181198quant_dense_181200quant_dense_181202*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_181191�
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0quant_dense_1_181254quant_dense_1_181256quant_dense_1_181258quant_dense_1_181260quant_dense_1_181262quant_dense_1_181264*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181253}
IdentityIdentity.quant_dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:&"
 
_user_specified_name181264:&"
 
_user_specified_name181262:&"
 
_user_specified_name181260:&"
 
_user_specified_name181258:&
"
 
_user_specified_name181256:&	"
 
_user_specified_name181254:&"
 
_user_specified_name181202:&"
 
_user_specified_name181200:&"
 
_user_specified_name181198:&"
 
_user_specified_name181196:&"
 
_user_specified_name181194:&"
 
_user_specified_name181192:&"
 
_user_specified_name181133:&"
 
_user_specified_name181131:^ Z
/
_output_shapes
:���������
'
_user_specified_nameflatten_input
�
�
$__inference_signature_wrapper_181563
flatten_input
unknown: 
	unknown_0: 
	unknown_1:
��
	unknown_2: 
	unknown_3: 
	unknown_4:	�
	unknown_5: 
	unknown_6: 
	unknown_7:	�

	unknown_8: 
	unknown_9: 

unknown_10:


unknown_11: 

unknown_12: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_181107o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181559:&"
 
_user_specified_name181557:&"
 
_user_specified_name181555:&"
 
_user_specified_name181553:&
"
 
_user_specified_name181551:&	"
 
_user_specified_name181549:&"
 
_user_specified_name181547:&"
 
_user_specified_name181545:&"
 
_user_specified_name181543:&"
 
_user_specified_name181541:&"
 
_user_specified_name181539:&"
 
_user_specified_name181537:&"
 
_user_specified_name181535:&"
 
_user_specified_name181533:^ Z
/
_output_shapes
:���������
'
_user_specified_nameflatten_input
ʌ
�
__inference__traced_save_182073
file_prefixB
8read_disablecopyonread_quantize_layer_quantize_layer_min: D
:read_1_disablecopyonread_quantize_layer_quantize_layer_max: @
6read_2_disablecopyonread_quantize_layer_optimizer_step: ?
5read_3_disablecopyonread_quant_flatten_optimizer_step: =
3read_4_disablecopyonread_quant_dense_optimizer_step: 9
/read_5_disablecopyonread_quant_dense_kernel_min: 9
/read_6_disablecopyonread_quant_dense_kernel_max: B
8read_7_disablecopyonread_quant_dense_post_activation_min: B
8read_8_disablecopyonread_quant_dense_post_activation_max: ?
5read_9_disablecopyonread_quant_dense_1_optimizer_step: <
2read_10_disablecopyonread_quant_dense_1_kernel_min: <
2read_11_disablecopyonread_quant_dense_1_kernel_max: D
:read_12_disablecopyonread_quant_dense_1_pre_activation_min: D
:read_13_disablecopyonread_quant_dense_1_pre_activation_max: :
&read_14_disablecopyonread_dense_kernel:
��3
$read_15_disablecopyonread_dense_bias:	�;
(read_16_disablecopyonread_dense_1_kernel:	�
4
&read_17_disablecopyonread_dense_1_bias:
-
#read_18_disablecopyonread_adam_iter:	 /
%read_19_disablecopyonread_adam_beta_1: /
%read_20_disablecopyonread_adam_beta_2: .
$read_21_disablecopyonread_adam_decay: 6
,read_22_disablecopyonread_adam_learning_rate: +
!read_23_disablecopyonread_total_1: +
!read_24_disablecopyonread_count_1: )
read_25_disablecopyonread_total: )
read_26_disablecopyonread_count: A
-read_27_disablecopyonread_adam_dense_kernel_m:
��:
+read_28_disablecopyonread_adam_dense_bias_m:	�B
/read_29_disablecopyonread_adam_dense_1_kernel_m:	�
;
-read_30_disablecopyonread_adam_dense_1_bias_m:
A
-read_31_disablecopyonread_adam_dense_kernel_v:
��:
+read_32_disablecopyonread_adam_dense_bias_v:	�B
/read_33_disablecopyonread_adam_dense_1_kernel_v:	�
;
-read_34_disablecopyonread_adam_dense_1_bias_v:

savev2_const
identity_71��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead8read_disablecopyonread_quantize_layer_quantize_layer_min"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp8read_disablecopyonread_quantize_layer_quantize_layer_min^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_1/DisableCopyOnReadDisableCopyOnRead:read_1_disablecopyonread_quantize_layer_quantize_layer_max"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp:read_1_disablecopyonread_quantize_layer_quantize_layer_max^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_2/DisableCopyOnReadDisableCopyOnRead6read_2_disablecopyonread_quantize_layer_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp6read_2_disablecopyonread_quantize_layer_optimizer_step^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_3/DisableCopyOnReadDisableCopyOnRead5read_3_disablecopyonread_quant_flatten_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp5read_3_disablecopyonread_quant_flatten_optimizer_step^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_4/DisableCopyOnReadDisableCopyOnRead3read_4_disablecopyonread_quant_dense_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp3read_4_disablecopyonread_quant_dense_optimizer_step^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_5/DisableCopyOnReadDisableCopyOnRead/read_5_disablecopyonread_quant_dense_kernel_min"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp/read_5_disablecopyonread_quant_dense_kernel_min^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_6/DisableCopyOnReadDisableCopyOnRead/read_6_disablecopyonread_quant_dense_kernel_max"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp/read_6_disablecopyonread_quant_dense_kernel_max^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_7/DisableCopyOnReadDisableCopyOnRead8read_7_disablecopyonread_quant_dense_post_activation_min"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp8read_7_disablecopyonread_quant_dense_post_activation_min^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead8read_8_disablecopyonread_quant_dense_post_activation_max"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp8read_8_disablecopyonread_quant_dense_post_activation_max^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead5read_9_disablecopyonread_quant_dense_1_optimizer_step"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp5read_9_disablecopyonread_quant_dense_1_optimizer_step^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead2read_10_disablecopyonread_quant_dense_1_kernel_min"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp2read_10_disablecopyonread_quant_dense_1_kernel_min^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnRead2read_11_disablecopyonread_quant_dense_1_kernel_max"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp2read_11_disablecopyonread_quant_dense_1_kernel_max^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead:read_12_disablecopyonread_quant_dense_1_pre_activation_min"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp:read_12_disablecopyonread_quant_dense_1_pre_activation_min^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_13/DisableCopyOnReadDisableCopyOnRead:read_13_disablecopyonread_quant_dense_1_pre_activation_max"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp:read_13_disablecopyonread_quant_dense_1_pre_activation_max^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_14/DisableCopyOnReadDisableCopyOnRead&read_14_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp&read_14_disablecopyonread_dense_kernel^Read_14/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_15/DisableCopyOnReadDisableCopyOnRead$read_15_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp$read_15_disablecopyonread_dense_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_dense_1_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
{
Read_17/DisableCopyOnReadDisableCopyOnRead&read_17_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp&read_17_disablecopyonread_dense_1_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:
x
Read_18/DisableCopyOnReadDisableCopyOnRead#read_18_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp#read_18_disablecopyonread_adam_iter^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_19/DisableCopyOnReadDisableCopyOnRead%read_19_disablecopyonread_adam_beta_1"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp%read_19_disablecopyonread_adam_beta_1^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_20/DisableCopyOnReadDisableCopyOnRead%read_20_disablecopyonread_adam_beta_2"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp%read_20_disablecopyonread_adam_beta_2^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_21/DisableCopyOnReadDisableCopyOnRead$read_21_disablecopyonread_adam_decay"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp$read_21_disablecopyonread_adam_decay^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_22/DisableCopyOnReadDisableCopyOnRead,read_22_disablecopyonread_adam_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp,read_22_disablecopyonread_adam_learning_rate^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_23/DisableCopyOnReadDisableCopyOnRead!read_23_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp!read_23_disablecopyonread_total_1^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_24/DisableCopyOnReadDisableCopyOnRead!read_24_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp!read_24_disablecopyonread_count_1^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_25/DisableCopyOnReadDisableCopyOnReadread_25_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpread_25_disablecopyonread_total^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_26/DisableCopyOnReadDisableCopyOnReadread_26_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpread_26_disablecopyonread_count^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_27/DisableCopyOnReadDisableCopyOnRead-read_27_disablecopyonread_adam_dense_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp-read_27_disablecopyonread_adam_dense_kernel_m^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_28/DisableCopyOnReadDisableCopyOnRead+read_28_disablecopyonread_adam_dense_bias_m"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp+read_28_disablecopyonread_adam_dense_bias_m^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_29/DisableCopyOnReadDisableCopyOnRead/read_29_disablecopyonread_adam_dense_1_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp/read_29_disablecopyonread_adam_dense_1_kernel_m^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
�
Read_30/DisableCopyOnReadDisableCopyOnRead-read_30_disablecopyonread_adam_dense_1_bias_m"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp-read_30_disablecopyonread_adam_dense_1_bias_m^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_31/DisableCopyOnReadDisableCopyOnRead-read_31_disablecopyonread_adam_dense_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp-read_31_disablecopyonread_adam_dense_kernel_v^Read_31/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_32/DisableCopyOnReadDisableCopyOnRead+read_32_disablecopyonread_adam_dense_bias_v"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp+read_32_disablecopyonread_adam_dense_bias_v^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_33/DisableCopyOnReadDisableCopyOnRead/read_33_disablecopyonread_adam_dense_1_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp/read_33_disablecopyonread_adam_dense_1_kernel_v^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
�
Read_34/DisableCopyOnReadDisableCopyOnRead-read_34_disablecopyonread_adam_dense_1_bias_v"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp-read_34_disablecopyonread_adam_dense_1_bias_v^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*�
value�B�$BBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-3/pre_activation_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-3/pre_activation_max/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *2
dtypes(
&2$	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_70Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_71IdentityIdentity_70:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_71Identity_71:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=$9

_output_shapes
: 

_user_specified_nameConst:3#/
-
_user_specified_nameAdam/dense_1/bias/v:5"1
/
_user_specified_nameAdam/dense_1/kernel/v:1!-
+
_user_specified_nameAdam/dense/bias/v:3 /
-
_user_specified_nameAdam/dense/kernel/v:3/
-
_user_specified_nameAdam/dense_1/bias/m:51
/
_user_specified_nameAdam/dense_1/kernel/m:1-
+
_user_specified_nameAdam/dense/bias/m:3/
-
_user_specified_nameAdam/dense/kernel/m:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:2.
,
_user_specified_nameAdam/learning_rate:*&
$
_user_specified_name
Adam/decay:+'
%
_user_specified_nameAdam/beta_2:+'
%
_user_specified_nameAdam/beta_1:)%
#
_user_specified_name	Adam/iter:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:@<
:
_user_specified_name" quant_dense_1/pre_activation_max:@<
:
_user_specified_name" quant_dense_1/pre_activation_min:84
2
_user_specified_namequant_dense_1/kernel_max:84
2
_user_specified_namequant_dense_1/kernel_min:<
8
6
_user_specified_namequant_dense_1/optimizer_step:?	;
9
_user_specified_name!quant_dense/post_activation_max:?;
9
_user_specified_name!quant_dense/post_activation_min:62
0
_user_specified_namequant_dense/kernel_max:62
0
_user_specified_namequant_dense/kernel_min::6
4
_user_specified_namequant_dense/optimizer_step:<8
6
_user_specified_namequant_flatten/optimizer_step:=9
7
_user_specified_namequantize_layer/optimizer_step:A=
;
_user_specified_name#!quantize_layer/quantize_layer_max:A=
;
_user_specified_name#!quantize_layer/quantize_layer_min:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ͤ
�
"__inference__traced_restore_182187
file_prefix<
2assignvariableop_quantize_layer_quantize_layer_min: >
4assignvariableop_1_quantize_layer_quantize_layer_max: :
0assignvariableop_2_quantize_layer_optimizer_step: 9
/assignvariableop_3_quant_flatten_optimizer_step: 7
-assignvariableop_4_quant_dense_optimizer_step: 3
)assignvariableop_5_quant_dense_kernel_min: 3
)assignvariableop_6_quant_dense_kernel_max: <
2assignvariableop_7_quant_dense_post_activation_min: <
2assignvariableop_8_quant_dense_post_activation_max: 9
/assignvariableop_9_quant_dense_1_optimizer_step: 6
,assignvariableop_10_quant_dense_1_kernel_min: 6
,assignvariableop_11_quant_dense_1_kernel_max: >
4assignvariableop_12_quant_dense_1_pre_activation_min: >
4assignvariableop_13_quant_dense_1_pre_activation_max: 4
 assignvariableop_14_dense_kernel:
��-
assignvariableop_15_dense_bias:	�5
"assignvariableop_16_dense_1_kernel:	�
.
 assignvariableop_17_dense_1_bias:
'
assignvariableop_18_adam_iter:	 )
assignvariableop_19_adam_beta_1: )
assignvariableop_20_adam_beta_2: (
assignvariableop_21_adam_decay: 0
&assignvariableop_22_adam_learning_rate: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: #
assignvariableop_25_total: #
assignvariableop_26_count: ;
'assignvariableop_27_adam_dense_kernel_m:
��4
%assignvariableop_28_adam_dense_bias_m:	�<
)assignvariableop_29_adam_dense_1_kernel_m:	�
5
'assignvariableop_30_adam_dense_1_bias_m:
;
'assignvariableop_31_adam_dense_kernel_v:
��4
%assignvariableop_32_adam_dense_bias_v:	�<
)assignvariableop_33_adam_dense_1_kernel_v:	�
5
'assignvariableop_34_adam_dense_1_bias_v:

identity_36��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*�
value�B�$BBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-3/pre_activation_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-3/pre_activation_max/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp2assignvariableop_quantize_layer_quantize_layer_minIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp4assignvariableop_1_quantize_layer_quantize_layer_maxIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_quantize_layer_optimizer_stepIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_quant_flatten_optimizer_stepIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp-assignvariableop_4_quant_dense_optimizer_stepIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp)assignvariableop_5_quant_dense_kernel_minIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp)assignvariableop_6_quant_dense_kernel_maxIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp2assignvariableop_7_quant_dense_post_activation_minIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp2assignvariableop_8_quant_dense_post_activation_maxIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp/assignvariableop_9_quant_dense_1_optimizer_stepIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp,assignvariableop_10_quant_dense_1_kernel_minIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp,assignvariableop_11_quant_dense_1_kernel_maxIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp4assignvariableop_12_quant_dense_1_pre_activation_minIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp4assignvariableop_13_quant_dense_1_pre_activation_maxIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_dense_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_1_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_1_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_mIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_mIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_vIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_vIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_vIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_vIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_36Identity_36:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:3#/
-
_user_specified_nameAdam/dense_1/bias/v:5"1
/
_user_specified_nameAdam/dense_1/kernel/v:1!-
+
_user_specified_nameAdam/dense/bias/v:3 /
-
_user_specified_nameAdam/dense/kernel/v:3/
-
_user_specified_nameAdam/dense_1/bias/m:51
/
_user_specified_nameAdam/dense_1/kernel/m:1-
+
_user_specified_nameAdam/dense/bias/m:3/
-
_user_specified_nameAdam/dense/kernel/m:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:2.
,
_user_specified_nameAdam/learning_rate:*&
$
_user_specified_name
Adam/decay:+'
%
_user_specified_nameAdam/beta_2:+'
%
_user_specified_nameAdam/beta_1:)%
#
_user_specified_name	Adam/iter:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:@<
:
_user_specified_name" quant_dense_1/pre_activation_max:@<
:
_user_specified_name" quant_dense_1/pre_activation_min:84
2
_user_specified_namequant_dense_1/kernel_max:84
2
_user_specified_namequant_dense_1/kernel_min:<
8
6
_user_specified_namequant_dense_1/optimizer_step:?	;
9
_user_specified_name!quant_dense/post_activation_max:?;
9
_user_specified_name!quant_dense/post_activation_min:62
0
_user_specified_namequant_dense/kernel_max:62
0
_user_specified_namequant_dense/kernel_min::6
4
_user_specified_namequant_dense/optimizer_step:<8
6
_user_specified_namequant_flatten/optimizer_step:=9
7
_user_specified_namequantize_layer/optimizer_step:A=
;
_user_specified_name#!quantize_layer/quantize_layer_max:A=
;
_user_specified_name#!quantize_layer/quantize_layer_min:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181633

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
.__inference_quant_dense_1_layer_call_fn_181754

inputs
unknown:	�

	unknown_0: 
	unknown_1: 
	unknown_2:

	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181253o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181750:&"
 
_user_specified_name181748:&"
 
_user_specified_name181746:&"
 
_user_specified_name181744:&"
 
_user_specified_name181742:&"
 
_user_specified_name181740:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_sequential_layer_call_and_return_conditional_losses_181361
flatten_input
quantize_layer_181280: 
quantize_layer_181282: &
quant_dense_181313:
��
quant_dense_181315: 
quant_dense_181317: !
quant_dense_181319:	�
quant_dense_181321: 
quant_dense_181323: '
quant_dense_1_181347:	�

quant_dense_1_181349: 
quant_dense_1_181351: "
quant_dense_1_181353:

quant_dense_1_181355: 
quant_dense_1_181357: 
identity��#quant_dense/StatefulPartitionedCall�%quant_dense_1/StatefulPartitionedCall�&quantize_layer/StatefulPartitionedCall�
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallflatten_inputquantize_layer_181280quantize_layer_181282*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181279�
quant_flatten/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181290�
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_181313quant_dense_181315quant_dense_181317quant_dense_181319quant_dense_181321quant_dense_181323*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_181312�
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0quant_dense_1_181347quant_dense_1_181349quant_dense_1_181351quant_dense_1_181353quant_dense_1_181355quant_dense_1_181357*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181346}
IdentityIdentity.quant_dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:&"
 
_user_specified_name181357:&"
 
_user_specified_name181355:&"
 
_user_specified_name181353:&"
 
_user_specified_name181351:&
"
 
_user_specified_name181349:&	"
 
_user_specified_name181347:&"
 
_user_specified_name181323:&"
 
_user_specified_name181321:&"
 
_user_specified_name181319:&"
 
_user_specified_name181317:&"
 
_user_specified_name181315:&"
 
_user_specified_name181313:&"
 
_user_specified_name181282:&"
 
_user_specified_name181280:^ Z
/
_output_shapes
:���������
'
_user_specified_nameflatten_input
�
�
/__inference_quantize_layer_layer_call_fn_181572

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181130w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181568:&"
 
_user_specified_name181566:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�"
�
G__inference_quant_dense_layer_call_and_return_conditional_losses_181312

inputsR
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
��J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: .
biasadd_readvariableop_resource:	�K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
��*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:�����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181141

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_quant_flatten_layer_call_fn_181616

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181141a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�Q
�
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181253

inputsB
/lastvaluequant_batchmin_readvariableop_resource:	�
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:
@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	�
*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������
y
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
,__inference_quant_dense_layer_call_fn_181667

inputs
unknown:
��
	unknown_0: 
	unknown_1: 
	unknown_2:	�
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_181312p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181663:&"
 
_user_specified_name181661:&"
 
_user_specified_name181659:&"
 
_user_specified_name181657:&"
 
_user_specified_name181655:&"
 
_user_specified_name181653:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
,__inference_quant_dense_layer_call_fn_181650

inputs
unknown:
��
	unknown_0: 
	unknown_1: 
	unknown_2:	�
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_181191p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181646:&"
 
_user_specified_name181644:&"
 
_user_specified_name181642:&"
 
_user_specified_name181640:&"
 
_user_specified_name181638:&"
 
_user_specified_name181636:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
.__inference_quant_dense_1_layer_call_fn_181771

inputs
unknown:	�

	unknown_0: 
	unknown_1: 
	unknown_2:

	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181346o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181767:&"
 
_user_specified_name181765:&"
 
_user_specified_name181763:&"
 
_user_specified_name181761:&"
 
_user_specified_name181759:&"
 
_user_specified_name181757:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181290

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181130

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity��#AllValuesQuantize/AssignMaxAllValue�#AllValuesQuantize/AssignMinAllValue�8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�(AllValuesQuantize/Maximum/ReadVariableOp�(AllValuesQuantize/Minimum/ReadVariableOpp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: r
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181602

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity��#AllValuesQuantize/AssignMaxAllValue�#AllValuesQuantize/AssignMinAllValue�8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�(AllValuesQuantize/Maximum/ReadVariableOp�(AllValuesQuantize/Minimum/ReadVariableOpp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: r
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:����������
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
O
flatten_input>
serving_default_flatten_input:0���������A
quant_dense_10
StatefulPartitionedCall:0���������
tensorflow/serving/predict:͸
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer
optimizer_step
 _weight_vars
!_quantize_activations
"_output_quantizers"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
	)layer
*optimizer_step
+_weight_vars
,
kernel_min
-
kernel_max
._quantize_activations
/post_activation_min
0post_activation_max
1_output_quantizers"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
	8layer
9optimizer_step
:_weight_vars
;
kernel_min
<
kernel_max
=_quantize_activations
>pre_activation_min
?pre_activation_max
@_output_quantizers"
_tf_keras_layer
�
0
1
2
3
A4
B5
*6
,7
-8
/9
010
C11
D12
913
;14
<15
>16
?17"
trackable_list_wrapper
<
A0
B1
C2
D3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
Jtrace_0
Ktrace_12�
+__inference_sequential_layer_call_fn_181394
+__inference_sequential_layer_call_fn_181427�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0zKtrace_1
�
Ltrace_0
Mtrace_12�
F__inference_sequential_layer_call_and_return_conditional_losses_181268
F__inference_sequential_layer_call_and_return_conditional_losses_181361�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0zMtrace_1
�B�
!__inference__wrapped_model_181107flatten_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_rateAm�Bm�Cm�Dm�Av�Bv�Cv�Dv�"
	optimizer
,
Sserving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ytrace_0
Ztrace_12�
/__inference_quantize_layer_layer_call_fn_181572
/__inference_quantize_layer_layer_call_fn_181581�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zYtrace_0zZtrace_1
�
[trace_0
\trace_12�
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181602
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181611�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0z\trace_1
):' 2!quantize_layer/quantize_layer_min
):' 2!quantize_layer/quantize_layer_max
:
min_var
max_var"
trackable_dict_wrapper
%:# 2quantize_layer/optimizer_step
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
btrace_0
ctrace_12�
.__inference_quant_flatten_layer_call_fn_181616
.__inference_quant_flatten_layer_call_fn_181621�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0zctrace_1
�
dtrace_0
etrace_12�
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181627
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181633�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0zetrace_1
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
$:" 2quant_flatten/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Q
A0
B1
*2
,3
-4
/5
06"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
qtrace_0
rtrace_12�
,__inference_quant_dense_layer_call_fn_181650
,__inference_quant_dense_layer_call_fn_181667�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0zrtrace_1
�
strace_0
ttrace_12�
G__inference_quant_dense_layer_call_and_return_conditional_losses_181716
G__inference_quant_dense_layer_call_and_return_conditional_losses_181737�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zstrace_0zttrace_1
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

Akernel
Bbias"
_tf_keras_layer
":  2quant_dense/optimizer_step
'
{0"
trackable_list_wrapper
: 2quant_dense/kernel_min
: 2quant_dense/kernel_max
 "
trackable_list_wrapper
':% 2quant_dense/post_activation_min
':% 2quant_dense/post_activation_max
 "
trackable_list_wrapper
Q
C0
D1
92
;3
<4
>5
?6"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quant_dense_1_layer_call_fn_181754
.__inference_quant_dense_1_layer_call_fn_181771�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181820
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181841�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ckernel
Dbias"
_tf_keras_layer
$:" 2quant_dense_1/optimizer_step
(
�0"
trackable_list_wrapper
 : 2quant_dense_1/kernel_min
 : 2quant_dense_1/kernel_max
 "
trackable_list_wrapper
(:& 2 quant_dense_1/pre_activation_min
(:& 2 quant_dense_1/pre_activation_max
 "
trackable_list_wrapper
 :
��2dense/kernel
:�2
dense/bias
!:	�
2dense_1/kernel
:
2dense_1/bias
�
0
1
2
3
*4
,5
-6
/7
08
99
;10
<11
>12
?13"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_sequential_layer_call_fn_181394flatten_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_sequential_layer_call_fn_181427flatten_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_181268flatten_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_181361flatten_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
$__inference_signature_wrapper_181563flatten_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 "

kwonlyargs�
jflatten_input
kwonlydefaults
 
annotations� *
 
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_quantize_layer_layer_call_fn_181572inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_quantize_layer_layer_call_fn_181581inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181602inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181611inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_flatten_layer_call_fn_181616inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_flatten_layer_call_fn_181621inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181627inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181633inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
C
*0
,1
-2
/3
04"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_quant_dense_layer_call_fn_181650inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_quant_dense_layer_call_fn_181667inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_quant_dense_layer_call_and_return_conditional_losses_181716inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_quant_dense_layer_call_and_return_conditional_losses_181737inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
B0"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
A0
�2"
trackable_tuple_wrapper
C
90
;1
<2
>3
?4"
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_dense_1_layer_call_fn_181754inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_dense_1_layer_call_fn_181771inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181820inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181841inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
D0"
trackable_list_wrapper
'
D0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
C0
�2"
trackable_tuple_wrapper
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
,min_var
-max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
;min_var
<max_var"
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
%:#
��2Adam/dense/kernel/m
:�2Adam/dense/bias/m
&:$	�
2Adam/dense_1/kernel/m
:
2Adam/dense_1/bias/m
%:#
��2Adam/dense/kernel/v
:�2Adam/dense/bias/v
&:$	�
2Adam/dense_1/kernel/v
:
2Adam/dense_1/bias/v�
!__inference__wrapped_model_181107�A,-B/0C;<D>?>�;
4�1
/�,
flatten_input���������
� "=�:
8
quant_dense_1'�$
quant_dense_1���������
�
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181820lC;<D>?4�1
*�'
!�
inputs����������
p
� ",�)
"�
tensor_0���������

� �
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_181841lC;<D>?4�1
*�'
!�
inputs����������
p 
� ",�)
"�
tensor_0���������

� �
.__inference_quant_dense_1_layer_call_fn_181754aC;<D>?4�1
*�'
!�
inputs����������
p
� "!�
unknown���������
�
.__inference_quant_dense_1_layer_call_fn_181771aC;<D>?4�1
*�'
!�
inputs����������
p 
� "!�
unknown���������
�
G__inference_quant_dense_layer_call_and_return_conditional_losses_181716mA,-B/04�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
G__inference_quant_dense_layer_call_and_return_conditional_losses_181737mA,-B/04�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
,__inference_quant_dense_layer_call_fn_181650bA,-B/04�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
,__inference_quant_dense_layer_call_fn_181667bA,-B/04�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181627l;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
tensor_0����������
� �
I__inference_quant_flatten_layer_call_and_return_conditional_losses_181633l;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
tensor_0����������
� �
.__inference_quant_flatten_layer_call_fn_181616a;�8
1�.
(�%
inputs���������
p
� ""�
unknown�����������
.__inference_quant_flatten_layer_call_fn_181621a;�8
1�.
(�%
inputs���������
p 
� ""�
unknown�����������
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181602w;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
J__inference_quantize_layer_layer_call_and_return_conditional_losses_181611w;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
/__inference_quantize_layer_layer_call_fn_181572l;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
/__inference_quantize_layer_layer_call_fn_181581l;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
F__inference_sequential_layer_call_and_return_conditional_losses_181268�A,-B/0C;<D>?F�C
<�9
/�,
flatten_input���������
p

 
� ",�)
"�
tensor_0���������

� �
F__inference_sequential_layer_call_and_return_conditional_losses_181361�A,-B/0C;<D>?F�C
<�9
/�,
flatten_input���������
p 

 
� ",�)
"�
tensor_0���������

� �
+__inference_sequential_layer_call_fn_181394{A,-B/0C;<D>?F�C
<�9
/�,
flatten_input���������
p

 
� "!�
unknown���������
�
+__inference_sequential_layer_call_fn_181427{A,-B/0C;<D>?F�C
<�9
/�,
flatten_input���������
p 

 
� "!�
unknown���������
�
$__inference_signature_wrapper_181563�A,-B/0C;<D>?O�L
� 
E�B
@
flatten_input/�,
flatten_input���������"=�:
8
quant_dense_1'�$
quant_dense_1���������
