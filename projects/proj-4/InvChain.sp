Lab 1 Problem 1A

* Bring in the library ... 
.lib 'cmoslibrary.lib' nominal

* My VCC is 
.param pvcc = 3

* Sizing Variables
.param alpha = 1.7

* Set Power and Ground as Global
.global vcc! gnd!

.subckt inv A Z 
  m1 Z A gnd! gnd! nmos w=1.4u l=0.35u AD=0.7p 
  m2 Z A vcc! vcc! pmos w=(1.4u*alpha) l=0.35u AD=0.7p*alpha  
.ends 

Cload z gnd! 30pF

Vin a gnd! 0V PWL 0 0NS 1NS 3 20NS 3

* Power Supplies
Vgnd gnd! 0 DC = 0
Vvcc vcc! 0 DC = 3V

* Analysis
.tran 1NS 100NS
.print tran v(a) v(z)

.OPTION MEASFORM=3

.OPTION POST
.TEMP 25 

.measure TRAN tphl_inv  TRIG v(Xinv1.a) VAL = 1.5 RISE = 1 TARG v(z) VAL=1.5 FALL = 1

.param fan = 7
Xinv1 a b inv M=1
Xinv2 b c inv M=fan**1
Xinv3 c d inv M=fan**2
Xinv4 d e inv M=fan**3
Xinv5 e f inv M=fan**4
Xinv6 f g inv M=fan**5
Xinv7 g h inv M=fan**6
Xinv8 h i inv M=fan**7
Xinv9 i j inv M=fan**8
Xinv10 j k inv M=fan**9
Xinv11 k z inv M=fan**11
.end