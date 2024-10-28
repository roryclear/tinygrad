#include <metal_stdlib>
using namespace metal;
kernel void r_128_16_8(device int* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup int temp0[16];
  int gidx0 = gid.x; /* 128 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (gidx0+(lidx0<<3)+-119);
  int alu1 = -((alu0<0)?0:alu0);
  *(temp0+lidx0) = -((-8<alu1)?alu1:-8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx0)!=1)) {
    int acc0 = 0;
    for (int ridx0 = 0; ridx0 < 16; ridx0++) {
      int val0 = *(temp0+ridx0);
      acc0 = (acc0+val0);
    }
    *(data0+gidx0) = (acc0+-1);
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void E_2048_32_3(device float* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 2048 */
  int lidx0 = lid.x; /* 32 */
  int alu0 = ((gidx0*96)+(lidx0*3));
  *(data0+alu0+1) = 0.0f;
  *(data0+alu0+2) = 0.0f;
  *(data0+alu0) = 0.0f;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_50257_50257(device int* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 50257 */
  *(data0+gidx0) = gidx0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_1024_16_64(device int* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup int temp0[16];
  int gidx0 = gid.x; /* 1024 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (gidx0+(lidx0<<6)+-959);
  int alu1 = -((alu0<0)?0:alu0);
  *(temp0+lidx0) = -((-64<alu1)?alu1:-64);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx0)!=1)) {
    int acc0 = 0;
    for (int ridx0 = 0; ridx0 < 16; ridx0++) {
      int val0 = *(temp0+ridx0);
      acc0 = (acc0+val0);
    }
    *(data0+gidx0) = (acc0+-1);
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_16_29_16_1733_3(device float* data0, device int* data1, device int* data2, device float* data3, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 29 */
  int gidx1 = gid.y; /* 16 */
  int lidx0 = lid.x; /* 16 */
  int gidx2 = gid.z; /* 13 */
  int val0 = *(data2+gidx2);
  int alu0 = ((gidx2*22272)+(gidx1*1392)+gidx0+(lidx0*87));
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  for (int ridx0 = 0; ridx0 < 1733; ridx0++) {
    int alu1 = ((gidx1*48)+(gidx0*1330944)+(lidx0*3)+(ridx0*768));
    int val1 = *(data1+(gidx0*1733)+ridx0);
    float cast0 = (float)(((val1!=val0)!=1));
    float val2 = *(data3+alu1+1);
    acc1 = (acc1+(cast0*val2));
    float val3 = *(data3+alu1+2);
    acc2 = (acc2+(cast0*val3));
    float val4 = *(data3+alu1);
    acc0 = (acc0+(cast0*val4));
  }
  *(data0+alu0+29) = acc1;
  *(data0+alu0+58) = acc2;
  *(data0+alu0) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_16_16_256_3_4(device float* data0, device int* data1, device int* data2, device float* data3, constant int& start_pos, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 16 */
  int lidx0 = lid.x; /* 16 */
  int gidx1 = gid.y; /* 13 */
  int val0 = *(data2+start_pos+gidx1);
  int alu0 = (lidx0*3);
  int alu1 = (gidx0*48);
  int alu2 = ((gidx1*768)+alu1+alu0);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  for (int ridx0 = 0; ridx0 < 256; ridx0++) {
    int alu3 = (alu1+alu0+(ridx0*3072));
    int alu4 = (ridx0<<2);
    int val1 = *(data1+alu4+1);
    float cast0 = (float)(((val1!=val0)!=1));
    int val2 = *(data1+alu4+2);
    float cast1 = (float)(((val2!=val0)!=1));
    int val3 = *(data1+alu4+3);
    float cast2 = (float)(((val3!=val0)!=1));
    int val4 = *(data1+alu4);
    float cast3 = (float)(((val4!=val0)!=1));
    float val5 = *(data3+alu3+1);
    float val6 = *(data3+alu3+2);
    float val7 = *(data3+alu3+768);
    float val8 = *(data3+alu3+769);
    float val9 = *(data3+alu3+770);
    float val10 = *(data3+alu3+1536);
    float val11 = *(data3+alu3+1537);
    float val12 = *(data3+alu3+1538);
    float val13 = *(data3+alu3+2304);
    float val14 = *(data3+alu3+2305);
    acc1 = (acc1+(cast3*val5)+(cast0*val8)+(cast1*val11)+(cast2*val14));
    float val15 = *(data3+alu3+2306);
    acc2 = (acc2+(cast3*val6)+(cast0*val9)+(cast1*val12)+(cast2*val15));
    float val16 = *(data3+alu3);
    acc0 = (acc0+(cast3*val16)+(cast0*val7)+(cast1*val10)+(cast2*val13));
  }
  *(data0+alu2+1) = acc1;
  *(data0+alu2+2) = acc2;
  *(data0+alu2) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_312_32_29(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 312 */
  int lidx0 = lid.x; /* 32 */
  int alu0 = ((gidx0*928)+(lidx0*29));
  int alu1 = ((gidx0<<5)+lidx0);
  float val0 = *(data1+alu0+1);
  float val1 = *(data1+alu0+2);
  float val2 = *(data1+alu0+3);
  float val3 = *(data1+alu0+4);
  float val4 = *(data1+alu0+5);
  float val5 = *(data1+alu0+6);
  float val6 = *(data1+alu0+7);
  float val7 = *(data1+alu0+8);
  float val8 = *(data1+alu0+9);
  float val9 = *(data1+alu0+10);
  float val10 = *(data1+alu0+11);
  float val11 = *(data1+alu0+12);
  float val12 = *(data1+alu0+13);
  float val13 = *(data1+alu0+14);
  float val14 = *(data1+alu0+15);
  float val15 = *(data1+alu0+16);
  float val16 = *(data1+alu0+17);
  float val17 = *(data1+alu0+18);
  float val18 = *(data1+alu0+19);
  float val19 = *(data1+alu0+20);
  float val20 = *(data1+alu0+21);
  float val21 = *(data1+alu0+22);
  float val22 = *(data1+alu0+23);
  float val23 = *(data1+alu0+24);
  float val24 = *(data1+alu0+25);
  float val25 = *(data1+alu0+26);
  float val26 = *(data1+alu0+27);
  float val27 = *(data1+alu0+28);
  float val28 = *(data1+alu0);
  float val29 = *(data2+alu1);
  *(data0+alu1) = (val28+val0+val1+val2+val3+val4+val5+val6+val7+val8+val9+val10+val11+val12+val13+val14+val15+val16+val17+val18+val19+val20+val21+val22+val23+val24+val25+val26+val27+val29);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_256_3(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup float temp0[256];
  int gidx0 = gid.x; /* 13 */
  int lidx0 = lid.x; /* 256 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float val0 = *(data1+(gidx0*768)+(lidx0*3)+ridx0);
    acc0 = (acc0+val0);
  }
  *(temp0+lidx0) = acc0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx0)!=1)) {
    float acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 256; ridx1++) {
      float val1 = *(temp0+ridx1);
      acc1 = (acc1+val1);
    }
    *(data0+gidx0) = (acc1*0.0013020833721384406f);
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_256_3n1(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup float temp0[256];
  int gidx0 = gid.x; /* 13 */
  int lidx0 = lid.x; /* 256 */
  float val0 = *(data2+gidx0);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float val1 = *(data1+(gidx0*768)+(lidx0*3)+ridx0);
    float alu0 = (val1-val0);
    acc0 = (acc0+(alu0*alu0));
  }
  *(temp0+lidx0) = acc0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx0)!=1)) {
    float acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 256; ridx1++) {
      float val2 = *(temp0+ridx1);
      acc1 = (acc1+val2);
    }
    *(data0+gidx0) = sqrt((1/((acc1*0.0013020833721384406f)+1e-05f)));
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void E_13_16_16_3(device float* data0, device float* data1, device float* data2, device float* data3, device float* data4, device float* data5, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 16 */
  int gidx1 = gid.y; /* 13 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (gidx0*48);
  int alu1 = (lidx0*3);
  int alu2 = ((gidx1*768)+alu0+alu1);
  int alu3 = (alu2+1);
  int alu4 = (alu2+2);
  int alu5 = (alu0+alu1);
  int alu6 = (alu5+1);
  int alu7 = (alu5+2);
  float val0 = *(data1+alu3);
  float val1 = *(data1+alu4);
  float val2 = *(data1+alu2);
  float val3 = *(data2+gidx1);
  float val4 = *(data3+gidx1);
  float val5 = *(data4+alu6);
  float val6 = *(data4+alu7);
  float val7 = *(data4+alu5);
  float val8 = *(data5+alu6);
  *(data0+alu3) = (((val0-val3)*val4*val5)+val8);
  float val9 = *(data5+alu7);
  *(data0+alu4) = (((val1-val3)*val4*val6)+val9);
  float val10 = *(data5+alu5);
  *(data0+alu2) = (((val2-val3)*val4*val7)+val10);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_48_16_192_3_4(device float* data0, device float* data1, device float* data2, device float* data3, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 48 */
  int gidx1 = gid.y; /* 13 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (lidx0*3);
  int alu1 = (gidx0*48);
  int alu2 = (alu1+alu0);
  int alu3 = ((gidx1*2304)+alu1+alu0);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  for (int ridx0 = 0; ridx0 < 192; ridx0++) {
    int alu4 = (alu2+(ridx0*9216));
    float val0 = *(data2+alu4+1);
    float val1 = *(data2+alu4+2);
    float val2 = *(data2+alu4+2304);
    float val3 = *(data2+alu4+2305);
    float val4 = *(data2+alu4+2306);
    float val5 = *(data2+alu4+4608);
    float val6 = *(data2+alu4+4609);
    float val7 = *(data2+alu4+4610);
    float val8 = *(data2+alu4+6912);
    float val9 = *(data2+alu4+6913);
    float val10 = *(data2+alu4+6914);
    float val11 = *(data2+alu4);
    float4 val12 = *((device float4*)(data1+(gidx1*768)+(ridx0<<2)));
    acc0 = (acc0+(val12.x*val11)+(val12.y*val2)+(val12.z*val5)+(val12.w*val8));
    acc1 = (acc1+(val12.x*val0)+(val12.y*val3)+(val12.z*val6)+(val12.w*val9));
    acc2 = (acc2+(val12.x*val1)+(val12.y*val4)+(val12.z*val7)+(val12.w*val10));
  }
  float val13 = *(data3+alu2+1);
  *(data0+alu3+1) = (acc1+val13);
  float val14 = *(data3+alu2+2);
  *(data0+alu3+2) = (acc2+val14);
  float val15 = *(data3+alu2);
  *(data0+alu3) = (acc0+val15);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_13_48_16_2(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 48 */
  int gidx1 = gid.y; /* 13 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (gidx0<<4);
  int alu1 = ((gidx1*768)+alu0+lidx0);
  int alu2 = ((gidx1*2304)+alu0+lidx0);
  float val0 = *(data1+alu2+768);
  *(data0+alu1) = val0;
  float val1 = *(data1+alu2+1536);
  *(data0+alu1+98304) = val1;
}
#include <metal_stdlib>
using namespace metal;
kernel void E_13_13(device float* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 13 */
  int gidx1 = gid.y; /* 13 */
  *(data0+(gidx1*13)+gidx0) = ((((((gidx1*49)+gidx0+49)%50)<25)?1:0)?-INFINITY:0.0f);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_13_4_16_3_4(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 13 */
  int gidx1 = gid.y; /* 13 */
  int lidx0 = lid.x; /* 4 */
  int alu0 = (lidx0*192);
  int alu1 = ((gidx1*13)+gidx0);
  int alu2 = (alu1+(lidx0*507));
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    int alu3 = (ridx0<<2);
    int alu4 = ((gidx0*2304)+alu0+alu3);
    int alu5 = ((gidx1*2304)+alu0+alu3);
    float4 val0 = *((device float4*)(data1+alu4+768));
    float4 val1 = *((device float4*)(data1+alu4+832));
    float4 val2 = *((device float4*)(data1+alu4+896));
    float4 val3 = *((device float4*)(data1+alu5+64));
    acc1 = (acc1+(val3.x*val1.x)+(val3.y*val1.y)+(val3.z*val1.z)+(val3.w*val1.w));
    float4 val4 = *((device float4*)(data1+alu5+128));
    acc2 = (acc2+(val4.x*val2.x)+(val4.y*val2.y)+(val4.z*val2.z)+(val4.w*val2.w));
    float4 val5 = *((device float4*)(data1+alu5));
    acc0 = (acc0+(val5.x*val0.x)+(val5.y*val0.y)+(val5.z*val0.z)+(val5.w*val0.w));
  }
  float val6 = *(data2+alu1);
  *(data0+alu2+169) = ((acc1*0.125f)+val6);
  *(data0+alu2+338) = ((acc2*0.125f)+val6);
  *(data0+alu2) = ((acc0*0.125f)+val6);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_39_4_13(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 39 */
  int lidx0 = lid.x; /* 4 */
  int alu0 = ((gidx0*52)+(lidx0*13));
  float val0 = *(data1+alu0+1);
  float val1 = *(data1+alu0+2);
  float val2 = *(data1+alu0+3);
  float val3 = *(data1+alu0+4);
  float val4 = *(data1+alu0+5);
  float val5 = *(data1+alu0+6);
  float val6 = *(data1+alu0+7);
  float val7 = *(data1+alu0+8);
  float val8 = *(data1+alu0+9);
  float val9 = *(data1+alu0+10);
  float val10 = *(data1+alu0+11);
  float val11 = *(data1+alu0+12);
  float val12 = *(data1+alu0);
  float alu1 = ((val12<val0)?val0:val12);
  float alu2 = ((alu1<val1)?val1:alu1);
  float alu3 = ((alu2<val2)?val2:alu2);
  float alu4 = ((alu3<val3)?val3:alu3);
  float alu5 = ((alu4<val4)?val4:alu4);
  float alu6 = ((alu5<val5)?val5:alu5);
  float alu7 = ((alu6<val6)?val6:alu6);
  float alu8 = ((alu7<val7)?val7:alu7);
  float alu9 = ((alu8<val8)?val8:alu8);
  float alu10 = ((alu9<val9)?val9:alu9);
  float alu11 = ((alu10<val10)?val10:alu10);
  *(data0+(gidx0<<2)+lidx0) = ((alu11<val11)?val11:alu11);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_39_4_13n1(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 39 */
  int lidx0 = lid.x; /* 4 */
  int alu0 = ((gidx0*52)+(lidx0*13));
  int alu1 = ((gidx0<<2)+lidx0);
  float val0 = *(data1+alu0+1);
  float val1 = *(data1+alu0+2);
  float val2 = *(data1+alu0+3);
  float val3 = *(data1+alu0+4);
  float val4 = *(data1+alu0+5);
  float val5 = *(data1+alu0+6);
  float val6 = *(data1+alu0+7);
  float val7 = *(data1+alu0+8);
  float val8 = *(data1+alu0+9);
  float val9 = *(data1+alu0+10);
  float val10 = *(data1+alu0+11);
  float val11 = *(data1+alu0+12);
  float val12 = *(data1+alu0);
  float val13 = *(data2+alu1);
  *(data0+alu1) = (exp2(((val12-val13)*1.4426950408889634f))+exp2(((val0-val13)*1.4426950408889634f))+exp2(((val1-val13)*1.4426950408889634f))+exp2(((val2-val13)*1.4426950408889634f))+exp2(((val3-val13)*1.4426950408889634f))+exp2(((val4-val13)*1.4426950408889634f))+exp2(((val5-val13)*1.4426950408889634f))+exp2(((val6-val13)*1.4426950408889634f))+exp2(((val7-val13)*1.4426950408889634f))+exp2(((val8-val13)*1.4426950408889634f))+exp2(((val9-val13)*1.4426950408889634f))+exp2(((val10-val13)*1.4426950408889634f))+exp2(((val11-val13)*1.4426950408889634f)));
}
#include <metal_stdlib>
using namespace metal;
kernel void E_13_13_4_3(device float* data0, device float* data1, device float* data2, device float* data3, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 13 */
  int gidx1 = gid.y; /* 13 */
  int lidx0 = lid.x; /* 4 */
  int alu0 = ((gidx1*12)+(lidx0*3));
  int alu1 = (alu0+1);
  int alu2 = (alu0+2);
  int alu3 = ((gidx1*156)+gidx0+(lidx0*39));
  int alu4 = (alu3+13);
  int alu5 = (alu3+26);
  float val0 = *(data1+alu4);
  float val1 = *(data1+alu5);
  float val2 = *(data1+alu3);
  float val3 = *(data2+alu1);
  float val4 = *(data2+alu2);
  float val5 = *(data2+alu0);
  float val6 = *(data3+alu1);
  *(data0+alu4) = (exp2(((val0-val3)*1.4426950408889634f))*(1/val6));
  float val7 = *(data3+alu2);
  *(data0+alu5) = (exp2(((val1-val4)*1.4426950408889634f))*(1/val7));
  float val8 = *(data3+alu0);
  *(data0+alu3) = (exp2(((val2-val5)*1.4426950408889634f))*(1/val8));
}
#include <metal_stdlib>
using namespace metal;
kernel void r_3_13_4_16_4_13(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 13 */
  int gidx1 = gid.y; /* 3 */
  int lidx0 = lid.x; /* 4 */
  int lidx1 = lid.y; /* 16 */
  int alu0 = ((gidx1*676)+(gidx0*13)+(lidx0*169));
  int alu1 = (lidx1<<2);
  int alu2 = ((gidx1<<8)+(lidx0<<6)+alu1);
  float val0 = *(data1+alu0+1);
  float val1 = *(data1+alu0+2);
  float val2 = *(data1+alu0+3);
  float val3 = *(data1+alu0+4);
  float val4 = *(data1+alu0+5);
  float val5 = *(data1+alu0+6);
  float val6 = *(data1+alu0+7);
  float val7 = *(data1+alu0+8);
  float val8 = *(data1+alu0+9);
  float val9 = *(data1+alu0+10);
  float val10 = *(data1+alu0+11);
  float val11 = *(data1+alu0+12);
  float val12 = *(data1+alu0);
  float4 val13 = *((device float4*)(data2+alu2+1536));
  float4 val14 = *((device float4*)(data2+alu2+3840));
  float4 val15 = *((device float4*)(data2+alu2+6144));
  float4 val16 = *((device float4*)(data2+alu2+8448));
  float4 val17 = *((device float4*)(data2+alu2+10752));
  float4 val18 = *((device float4*)(data2+alu2+13056));
  float4 val19 = *((device float4*)(data2+alu2+15360));
  float4 val20 = *((device float4*)(data2+alu2+17664));
  float4 val21 = *((device float4*)(data2+alu2+19968));
  float4 val22 = *((device float4*)(data2+alu2+22272));
  float4 val23 = *((device float4*)(data2+alu2+24576));
  float4 val24 = *((device float4*)(data2+alu2+26880));
  float4 val25 = *((device float4*)(data2+alu2+29184));
  *((device float4*)(data0+(gidx1*3328)+(gidx0<<6)+(lidx0*832)+alu1)) = float4(((val12*val13.x)+(val0*val14.x)+(val1*val15.x)+(val2*val16.x)+(val3*val17.x)+(val4*val18.x)+(val5*val19.x)+(val6*val20.x)+(val7*val21.x)+(val8*val22.x)+(val9*val23.x)+(val10*val24.x)+(val11*val25.x)),((val12*val13.y)+(val0*val14.y)+(val1*val15.y)+(val2*val16.y)+(val3*val17.y)+(val4*val18.y)+(val5*val19.y)+(val6*val20.y)+(val7*val21.y)+(val8*val22.y)+(val9*val23.y)+(val10*val24.y)+(val11*val25.y)),((val12*val13.z)+(val0*val14.z)+(val1*val15.z)+(val2*val16.z)+(val3*val17.z)+(val4*val18.z)+(val5*val19.z)+(val6*val20.z)+(val7*val21.z)+(val8*val22.z)+(val9*val23.z)+(val10*val24.z)+(val11*val25.z)),((val12*val13.w)+(val0*val14.w)+(val1*val15.w)+(val2*val16.w)+(val3*val17.w)+(val4*val18.w)+(val5*val19.w)+(val6*val20.w)+(val7*val21.w)+(val8*val22.w)+(val9*val23.w)+(val10*val24.w)+(val11*val25.w)));
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_16_16_192_3_4(device float* data0, device float* data1, device float* data2, device float* data3, device float* data4, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 16 */
  int gidx1 = gid.y; /* 13 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (lidx0*3);
  int alu1 = (gidx0*48);
  int alu2 = (alu1+alu0);
  int alu3 = ((gidx1*768)+alu1+alu0);
  int alu4 = (alu3+1);
  int alu5 = (alu3+2);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  for (int ridx0 = 0; ridx0 < 192; ridx0++) {
    int alu6 = (alu2+(ridx0*3072));
    int alu7 = (ridx0<<2);
    int alu8 = ((gidx1<<6)+((ridx0>>4)*832));
    float val0 = *(data2+alu8+((alu7+1)&63));
    float val1 = *(data2+alu8+((alu7+2)&63));
    float val2 = *(data2+alu8+((alu7+3)&63));
    float val3 = *(data2+alu8+(alu7&63));
    float val4 = *(data3+alu6+1);
    float val5 = *(data3+alu6+2);
    float val6 = *(data3+alu6+768);
    float val7 = *(data3+alu6+769);
    float val8 = *(data3+alu6+770);
    float val9 = *(data3+alu6+1536);
    float val10 = *(data3+alu6+1537);
    float val11 = *(data3+alu6+1538);
    float val12 = *(data3+alu6+2304);
    float val13 = *(data3+alu6+2305);
    acc1 = (acc1+(val3*val4)+(val0*val7)+(val1*val10)+(val2*val13));
    float val14 = *(data3+alu6+2306);
    acc2 = (acc2+(val3*val5)+(val0*val8)+(val1*val11)+(val2*val14));
    float val15 = *(data3+alu6);
    acc0 = (acc0+(val3*val15)+(val0*val6)+(val1*val9)+(val2*val12));
  }
  float val16 = *(data1+alu4);
  float val17 = *(data1+alu5);
  float val18 = *(data1+alu3);
  float val19 = *(data4+alu2+1);
  *(data0+alu4) = (val16+acc1+val19);
  float val20 = *(data4+alu2+2);
  *(data0+alu5) = (val17+acc2+val20);
  float val21 = *(data4+alu2);
  *(data0+alu3) = (val18+acc0+val21);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_64_16_192_3_4(device float* data0, device float* data1, device float* data2, device float* data3, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 64 */
  int gidx1 = gid.y; /* 13 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (lidx0*3);
  int alu1 = (gidx0*48);
  int alu2 = (alu1+alu0);
  int alu3 = ((gidx1*3072)+alu1+alu0);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  for (int ridx0 = 0; ridx0 < 192; ridx0++) {
    int alu4 = (alu2+(ridx0*12288));
    float val0 = *(data2+alu4+1);
    float val1 = *(data2+alu4+2);
    float val2 = *(data2+alu4+3072);
    float val3 = *(data2+alu4+3073);
    float val4 = *(data2+alu4+3074);
    float val5 = *(data2+alu4+6144);
    float val6 = *(data2+alu4+6145);
    float val7 = *(data2+alu4+6146);
    float val8 = *(data2+alu4+9216);
    float val9 = *(data2+alu4+9217);
    float val10 = *(data2+alu4+9218);
    float val11 = *(data2+alu4);
    float4 val12 = *((device float4*)(data1+(gidx1*768)+(ridx0<<2)));
    acc0 = (acc0+(val12.x*val11)+(val12.y*val2)+(val12.z*val5)+(val12.w*val8));
    acc1 = (acc1+(val12.x*val0)+(val12.y*val3)+(val12.z*val6)+(val12.w*val9));
    acc2 = (acc2+(val12.x*val1)+(val12.y*val4)+(val12.z*val7)+(val12.w*val10));
  }
  float val13 = *(data3+alu2+1);
  float alu5 = (acc1+val13);
  *(data0+alu3+1) = ((1/(exp2(((alu5+(alu5*alu5*alu5*0.044715f))*-2.302208198144325f))+1.0f))*alu5);
  float val14 = *(data3+alu2+2);
  float alu6 = (acc2+val14);
  *(data0+alu3+2) = ((1/(exp2(((alu6+(alu6*alu6*alu6*0.044715f))*-2.302208198144325f))+1.0f))*alu6);
  float val15 = *(data3+alu2);
  float alu7 = (acc0+val15);
  *(data0+alu3) = ((1/(exp2(((alu7+(alu7*alu7*alu7*0.044715f))*-2.302208198144325f))+1.0f))*alu7);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_16_16_768_3_4(device float* data0, device float* data1, device float* data2, device float* data3, device float* data4, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 16 */
  int gidx1 = gid.y; /* 13 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (lidx0*3);
  int alu1 = (gidx0*48);
  int alu2 = (alu1+alu0);
  int alu3 = ((gidx1*768)+alu1+alu0);
  int alu4 = (alu3+1);
  int alu5 = (alu3+2);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  for (int ridx0 = 0; ridx0 < 768; ridx0++) {
    int alu6 = (alu2+(ridx0*3072));
    float val0 = *(data3+alu6+1);
    float val1 = *(data3+alu6+2);
    float val2 = *(data3+alu6+768);
    float val3 = *(data3+alu6+769);
    float val4 = *(data3+alu6+770);
    float val5 = *(data3+alu6+1536);
    float val6 = *(data3+alu6+1537);
    float val7 = *(data3+alu6+1538);
    float val8 = *(data3+alu6+2304);
    float val9 = *(data3+alu6+2305);
    float val10 = *(data3+alu6+2306);
    float val11 = *(data3+alu6);
    float4 val12 = *((device float4*)(data2+(gidx1*3072)+(ridx0<<2)));
    acc0 = (acc0+(val12.x*val11)+(val12.y*val2)+(val12.z*val5)+(val12.w*val8));
    acc1 = (acc1+(val12.x*val0)+(val12.y*val3)+(val12.z*val6)+(val12.w*val9));
    acc2 = (acc2+(val12.x*val1)+(val12.y*val4)+(val12.z*val7)+(val12.w*val10));
  }
  float val13 = *(data1+alu4);
  float val14 = *(data1+alu5);
  float val15 = *(data1+alu3);
  float val16 = *(data4+alu2+1);
  *(data0+alu4) = (val13+acc1+val16);
  float val17 = *(data4+alu2+2);
  *(data0+alu5) = (val14+acc2+val17);
  float val18 = *(data4+alu2);
  *(data0+alu3) = (val15+acc0+val18);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_(device unsigned int* data0, device unsigned int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  *(data0+0) = (val0+786432u);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n1(device unsigned int* data0, device unsigned int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  *(data0+0) = (val0+1769472u);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n2(device unsigned int* data0, device unsigned int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  *(data0+0) = (val0+2304u);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n3(device unsigned int* data0, device unsigned int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  *(data0+0) = (val0+589824u);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n4(device unsigned int* data0, device unsigned int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  *(data0+0) = (val0+768u);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n5(device unsigned int* data0, device unsigned int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  *(data0+0) = (val0+2359296u);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n6(device unsigned int* data0, device unsigned int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  *(data0+0) = (val0+3072u);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_13_50257_192_4(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 50257 */
  int gidx1 = gid.y; /* 13 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 192; ridx0++) {
    int alu0 = (ridx0<<2);
    float4 val0 = *((device float4*)(data1+(gidx1*768)+alu0));
    float4 val1 = *((device float4*)(data2+(gidx0*768)+alu0));
    acc0 = (acc0+(val0.x*val1.x)+(val0.y*val1.y)+(val0.z*val1.z)+(val0.w*val1.w));
  }
  *(data0+(gidx1*50257)+gidx0) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_29_1733(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 29 */
  float acc0 = -INFINITY;
  for (int ridx0 = 0; ridx0 < 1733; ridx0++) {
    float val0 = *(data1+(gidx0*1733)+ridx0+603084);
    float alu0 = (val0*1.25f);
    acc0 = ((acc0<alu0)?alu0:acc0);
  }
  *(data0+gidx0) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_29(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  float val0 = *(data1+28);
  float4 val1 = *((device float4*)(data1+0));
  float alu0 = ((val1.x<val1.y)?val1.y:val1.x);
  float alu1 = ((alu0<val1.z)?val1.z:alu0);
  float alu2 = ((alu1<val1.w)?val1.w:alu1);
  float4 val2 = *((device float4*)(data1+4));
  float alu3 = ((alu2<val2.x)?val2.x:alu2);
  float alu4 = ((alu3<val2.y)?val2.y:alu3);
  float alu5 = ((alu4<val2.z)?val2.z:alu4);
  float alu6 = ((alu5<val2.w)?val2.w:alu5);
  float4 val3 = *((device float4*)(data1+8));
  float alu7 = ((alu6<val3.x)?val3.x:alu6);
  float alu8 = ((alu7<val3.y)?val3.y:alu7);
  float alu9 = ((alu8<val3.z)?val3.z:alu8);
  float alu10 = ((alu9<val3.w)?val3.w:alu9);
  float4 val4 = *((device float4*)(data1+12));
  float alu11 = ((alu10<val4.x)?val4.x:alu10);
  float alu12 = ((alu11<val4.y)?val4.y:alu11);
  float alu13 = ((alu12<val4.z)?val4.z:alu12);
  float alu14 = ((alu13<val4.w)?val4.w:alu13);
  float4 val5 = *((device float4*)(data1+16));
  float alu15 = ((alu14<val5.x)?val5.x:alu14);
  float alu16 = ((alu15<val5.y)?val5.y:alu15);
  float alu17 = ((alu16<val5.z)?val5.z:alu16);
  float alu18 = ((alu17<val5.w)?val5.w:alu17);
  float4 val6 = *((device float4*)(data1+20));
  float alu19 = ((alu18<val6.x)?val6.x:alu18);
  float alu20 = ((alu19<val6.y)?val6.y:alu19);
  float alu21 = ((alu20<val6.z)?val6.z:alu20);
  float alu22 = ((alu21<val6.w)?val6.w:alu21);
  float4 val7 = *((device float4*)(data1+24));
  float alu23 = ((alu22<val7.x)?val7.x:alu22);
  float alu24 = ((alu23<val7.y)?val7.y:alu23);
  float alu25 = ((alu24<val7.z)?val7.z:alu24);
  float alu26 = ((alu25<val7.w)?val7.w:alu25);
  *(data0+0) = ((alu26<val0)?val0:alu26);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_29_1733n1(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 29 */
  float val0 = *(data2+0);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 1733; ridx0++) {
    float val1 = *(data1+(gidx0*1733)+ridx0+603084);
    acc0 = (acc0+exp2((((val1*1.25f)-val0)*1.4426950408889634f)));
  }
  *(data0+gidx0) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_29n1(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  float val0 = *(data1+28);
  float4 val1 = *((device float4*)(data1+0));
  float4 val2 = *((device float4*)(data1+4));
  float4 val3 = *((device float4*)(data1+8));
  float4 val4 = *((device float4*)(data1+12));
  float4 val5 = *((device float4*)(data1+16));
  float4 val6 = *((device float4*)(data1+20));
  float4 val7 = *((device float4*)(data1+24));
  *(data0+0) = (val1.x+val1.y+val1.z+val1.w+val2.x+val2.y+val2.z+val2.w+val3.x+val3.y+val3.z+val3.w+val4.x+val4.y+val4.z+val4.w+val5.x+val5.y+val5.z+val5.w+val6.x+val6.y+val6.z+val6.w+val7.x+val7.y+val7.z+val7.w+val0);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_50257(device float* data0, device float* data1, device float* data2, device float* data3, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 50257 */
  float val0 = *(data1+gidx0+603084);
  float val1 = *(data2+0);
  float val2 = *(data3+0);
  *(data0+gidx0) = (exp2((((val0*1.25f)-val1)*1.4426950408889634f))*(1/val2));
}
#include <metal_stdlib>
using namespace metal;
kernel void r_197_16_16_64_4(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 16 */
  int gidx1 = gid.y; /* 197 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (gidx0<<4);
  int alu1 = ((gidx1<<8)+alu0+lidx0);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 64; ridx0++) {
    int alu2 = (ridx0<<2);
    int alu3 = (alu1+alu2);
    int alu4 = (alu0+lidx0+alu2);
    int alu5 = ((alu3+50177)%50432);
    int alu6 = ((alu3+50178)%50432);
    int alu7 = ((alu3+50179)%50432);
    int alu8 = ((alu3+50180)%50432);
    float val0 = ((((alu4<255)!=1)&((alu5<175)!=1))?*(data1+alu5+-175):0.0f);
    float val1 = ((((alu4<254)!=1)&((alu6<175)!=1))?*(data1+alu6+-175):0.0f);
    float val2 = ((((alu4<253)!=1)&((alu7<175)!=1))?*(data1+alu7+-175):0.0f);
    float val3 = ((((alu4<252)!=1)&((alu8<175)!=1))?*(data1+alu8+-175):0.0f);
    acc0 = (acc0+val0+val1+val2+val3);
  }
  *(data0+alu1) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_197_197(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 197 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 197; ridx0++) {
    float val0 = ((((gidx0+ridx0)<197)!=1)?*(data1+(gidx0<<8)+(ridx0<<8)+-50177):0.0f);
    acc0 = (acc0+val0);
  }
  *(data0+gidx0) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n7(device unsigned int* data0, device unsigned int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  *(data0+0) = (val0+38597376u);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n8(device unsigned int* data0, device unsigned int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  *(data0+0) = (val0+1u);
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n9(device unsigned int* data0, device unsigned int* data1, device unsigned long* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data1+0);
  unsigned long alu0 = (((unsigned long)((val0+1u))<<32)|(unsigned long)(val0));
  unsigned int cast0 = (unsigned int)(((alu0>>32)&4294967295ull));
  unsigned long val1 = *(data2+0);
  unsigned int cast1 = (unsigned int)((val1&4294967295ull));
  unsigned int cast2 = (unsigned int)(((val1>>32)&4294967295ull));
  unsigned int alu1 = (cast0+cast2);
  unsigned int alu2 = ((unsigned int)((alu0&4294967295ull))+cast1+alu1);
  unsigned int alu3 = (cast1^cast2^466688986u);
  unsigned int alu4 = (alu2^((cast0<<13)+(cast2<<13)+(alu1>>19)));
  unsigned int alu5 = (alu2+alu4);
  unsigned int alu6 = (alu5^((alu4<<15)+(alu4>>17)));
  unsigned int alu7 = (alu5+alu6);
  unsigned int alu8 = (alu7^((alu6<<26)+(alu6>>6)));
  unsigned int alu9 = (alu7+alu8);
  unsigned int alu10 = (alu9^((alu8<<6)+(alu8>>26)));
  unsigned int alu11 = (alu10+alu3);
  unsigned int alu12 = (alu11+alu9+cast2);
  unsigned int alu13 = ((alu12+1u)^((alu10<<17)+(alu3<<17)+((alu11+1u)>>15)+131072u));
  unsigned int alu14 = (alu12+alu13);
  unsigned int alu15 = ((alu14+1u)^((alu13<<29)+(alu13>>3)));
  unsigned int alu16 = (alu14+alu15);
  unsigned int alu17 = ((alu16+1u)^((alu15<<16)+(alu15>>16)));
  unsigned int alu18 = (alu16+alu17);
  unsigned int alu19 = ((alu18+1u)^((alu17<<24)+(alu17>>8)));
  unsigned int alu20 = (alu19+cast1);
  unsigned int alu21 = (alu20+alu18+alu3);
  unsigned int alu22 = ((alu21+3u)^((alu19<<13)+(cast1<<13)+((alu20+2u)>>19)+16384u));
  unsigned int alu23 = (alu21+alu22);
  unsigned int alu24 = ((alu23+3u)^((alu22<<15)+(alu22>>17)));
  unsigned int alu25 = (alu23+alu24);
  unsigned int alu26 = ((alu25+3u)^((alu24<<26)+(alu24>>6)));
  unsigned int alu27 = (alu25+alu26);
  unsigned int alu28 = ((alu27+3u)^((alu26<<6)+(alu26>>26)));
  unsigned int alu29 = (alu28+cast2);
  unsigned int alu30 = (alu29+alu27+cast1);
  unsigned int alu31 = ((alu30+6u)^((alu28<<17)+(cast2<<17)+((alu29+3u)>>15)+393216u));
  unsigned int alu32 = (alu30+alu31);
  unsigned int alu33 = ((alu32+6u)^((alu31<<29)+(alu31>>3)));
  unsigned int alu34 = (alu32+alu33);
  unsigned int alu35 = ((alu34+6u)^((alu33<<16)+(alu33>>16)));
  unsigned int alu36 = (alu34+alu35);
  unsigned int alu37 = ((alu36+6u)^((alu35<<24)+(alu35>>8)));
  unsigned int alu38 = (alu37+alu3);
  unsigned int alu39 = (alu38+alu36+cast2);
  unsigned int alu40 = ((alu39+10u)^((alu37<<13)+(alu3<<13)+((alu38+4u)>>19)+32768u));
  unsigned int alu41 = (alu39+alu40);
  unsigned int alu42 = ((alu41+10u)^((alu40<<15)+(alu40>>17)));
  unsigned int alu43 = (alu41+alu42);
  unsigned int alu44 = ((alu43+10u)^((alu42<<26)+(alu42>>6)));
  unsigned int alu45 = (alu43+alu44);
  *(data0+0) = (unsigned int)((((((unsigned long)((((alu45+10u)^((alu44<<6)+(alu44>>26)))+cast1+5u))<<32)|(unsigned long)((alu45+alu3+10u)))>>32)&4294967295ull));
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n10(device float* data0, device unsigned int* data1, device unsigned long* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  float precast0 = 1.0f;
  unsigned int val0 = *(data1+0);
  unsigned long alu0 = (((unsigned long)((val0+1u))<<32)|(unsigned long)(val0));
  unsigned int cast0 = (unsigned int)(((alu0>>32)&4294967295ull));
  unsigned long val1 = *(data2+0);
  unsigned int cast1 = (unsigned int)((val1&4294967295ull));
  unsigned int cast2 = (unsigned int)(((val1>>32)&4294967295ull));
  unsigned int alu1 = (cast0+cast2);
  unsigned int alu2 = ((unsigned int)((alu0&4294967295ull))+cast1+alu1);
  unsigned int alu3 = (cast1^cast2^466688986u);
  unsigned int alu4 = (alu2^((cast0<<13)+(cast2<<13)+(alu1>>19)));
  unsigned int alu5 = (alu2+alu4);
  unsigned int alu6 = (alu5^((alu4<<15)+(alu4>>17)));
  unsigned int alu7 = (alu5+alu6);
  unsigned int alu8 = (alu7^((alu6<<26)+(alu6>>6)));
  unsigned int alu9 = (alu7+alu8);
  unsigned int alu10 = (alu9^((alu8<<6)+(alu8>>26)));
  unsigned int alu11 = (alu10+alu3);
  unsigned int alu12 = (alu11+alu9+cast2);
  unsigned int alu13 = ((alu12+1u)^((alu10<<17)+(alu3<<17)+((alu11+1u)>>15)+131072u));
  unsigned int alu14 = (alu12+alu13);
  unsigned int alu15 = ((alu14+1u)^((alu13<<29)+(alu13>>3)));
  unsigned int alu16 = (alu14+alu15);
  unsigned int alu17 = ((alu16+1u)^((alu15<<16)+(alu15>>16)));
  unsigned int alu18 = (alu16+alu17);
  unsigned int alu19 = ((alu18+1u)^((alu17<<24)+(alu17>>8)));
  unsigned int alu20 = (alu19+cast1);
  unsigned int alu21 = (alu20+alu18+alu3);
  unsigned int alu22 = ((alu21+3u)^((alu19<<13)+(cast1<<13)+((alu20+2u)>>19)+16384u));
  unsigned int alu23 = (alu21+alu22);
  unsigned int alu24 = ((alu23+3u)^((alu22<<15)+(alu22>>17)));
  unsigned int alu25 = (alu23+alu24);
  unsigned int alu26 = ((alu25+3u)^((alu24<<26)+(alu24>>6)));
  unsigned int alu27 = (alu25+alu26);
  unsigned int alu28 = ((alu27+3u)^((alu26<<6)+(alu26>>26)));
  unsigned int alu29 = (alu28+cast2);
  unsigned int alu30 = (alu29+alu27+cast1);
  unsigned int alu31 = ((alu30+6u)^((alu28<<17)+(cast2<<17)+((alu29+3u)>>15)+393216u));
  unsigned int alu32 = (alu30+alu31);
  unsigned int alu33 = ((alu32+6u)^((alu31<<29)+(alu31>>3)));
  unsigned int alu34 = (alu32+alu33);
  unsigned int alu35 = ((alu34+6u)^((alu33<<16)+(alu33>>16)));
  unsigned int alu36 = (alu34+alu35);
  unsigned int alu37 = ((alu36+6u)^((alu35<<24)+(alu35>>8)));
  unsigned int alu38 = (alu37+alu3);
  unsigned int alu39 = (alu38+alu36+cast2);
  unsigned int alu40 = ((alu39+10u)^((alu37<<13)+(alu3<<13)+((alu38+4u)>>19)+32768u));
  unsigned int alu41 = (alu39+alu40);
  unsigned int alu42 = ((alu41+10u)^((alu40<<15)+(alu40>>17)));
  unsigned int alu43 = (alu41+alu42);
  unsigned int alu44 = ((alu43+10u)^((alu42<<26)+(alu42>>6)));
  unsigned int alu45 = (alu43+alu44);
  unsigned int precast1 = (((unsigned int)(((((unsigned long)((((alu45+10u)^((alu44<<6)+(alu44>>26)))+cast1+5u))<<32)|(unsigned long)((alu45+alu3+10u)))&4294967295ull))>>9)|as_type<unsigned int>(precast0));
  *(data0+0) = (as_type<float>(precast1)+-1.0f);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_29_1733n2(device int* data0, device float* data1, device float* data2, device float* data3, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 29 */
  float val0 = *(data1+0);
  float val1 = *(data3+196);
  float val2 = *(data2+50431);
  int acc0 = 0;
  for (int ridx0 = 0; ridx0 < 1733; ridx0++) {
    int alu0 = ((gidx0*1733)+ridx0+175);
    float val3 = *(data2+alu0);
    float val4 = *(data3+(alu0>>8));
    acc0 = (acc0+(int)(((val0<((val3+val4)*(1/(val2+val1))))!=1)));
  }
  *(data0+gidx0) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_29n2(device int* data0, device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int val0 = *(data1+0);
  int val1 = *(data1+1);
  int val2 = *(data1+2);
  int val3 = *(data1+3);
  int val4 = *(data1+4);
  int val5 = *(data1+5);
  int val6 = *(data1+6);
  int val7 = *(data1+7);
  int val8 = *(data1+8);
  int val9 = *(data1+9);
  int val10 = *(data1+10);
  int val11 = *(data1+11);
  int val12 = *(data1+12);
  int val13 = *(data1+13);
  int val14 = *(data1+14);
  int val15 = *(data1+15);
  int val16 = *(data1+16);
  int val17 = *(data1+17);
  int val18 = *(data1+18);
  int val19 = *(data1+19);
  int val20 = *(data1+20);
  int val21 = *(data1+21);
  int val22 = *(data1+22);
  int val23 = *(data1+23);
  int val24 = *(data1+24);
  int val25 = *(data1+25);
  int val26 = *(data1+26);
  int val27 = *(data1+27);
  int val28 = *(data1+28);
  *(data0+0) = (val0+val1+val2+val3+val4+val5+val6+val7+val8+val9+val10+val11+val12+val13+val14+val15+val16+val17+val18+val19+val20+val21+val22+val23+val24+val25+val26+val27+val28);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_768_16_64(device float* data0, device float* data1, device int* data2, device int* data3, device float* data4, constant int& start_pos, constant int& tokens, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup float temp0[16];
  int gidx0 = gid.x; /* 768 */
  int lidx0 = lid.x; /* 16 */
  int val0 = *(data3+start_pos);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 64; ridx0++) {
    int val1 = *(data2+(lidx0<<6)+ridx0);
    float val2 = *(data4+gidx0+(lidx0*49152)+(ridx0*768));
    acc0 = (acc0+((float)(((val1!=val0)!=1))*val2));
  }
  *(temp0+lidx0) = acc0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx0)!=1)) {
    float val3 = *(data1+(tokens*768)+gidx0);
    float acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 16; ridx1++) {
      float val4 = *(temp0+ridx1);
      acc1 = (acc1+val4);
    }
    *(data0+gidx0) = (val3+acc1);
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void r_256_3(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup float temp0[256];
  int lidx0 = lid.x; /* 256 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float val0 = *(data1+(lidx0*3)+ridx0);
    acc0 = (acc0+val0);
  }
  *(temp0+lidx0) = acc0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx0)!=1)) {
    float acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 256; ridx1++) {
      float val1 = *(temp0+ridx1);
      acc1 = (acc1+val1);
    }
    *(data0+0) = (acc1*0.0013020833721384406f);
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void r_256_3n1(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup float temp0[256];
  int lidx0 = lid.x; /* 256 */
  float val0 = *(data2+0);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float val1 = *(data1+(lidx0*3)+ridx0);
    float alu0 = (val1-val0);
    acc0 = (acc0+(alu0*alu0));
  }
  *(temp0+lidx0) = acc0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx0)!=1)) {
    float acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 256; ridx1++) {
      float val2 = *(temp0+ridx1);
      acc1 = (acc1+val2);
    }
    *(data0+0) = sqrt((1/((acc1*0.0013020833721384406f)+1e-05f)));
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void E_6_32_4(device float* data0, device float* data1, device float* data2, device float* data3, device float* data4, device float* data5, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 6 */
  int lidx0 = lid.x; /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  float val0 = *(data2+0);
  float val1 = *(data3+0);
  float4 val2 = *((device float4*)(data1+alu0));
  float4 val3 = *((device float4*)(data4+alu0));
  float4 val4 = *((device float4*)(data5+alu0));
  *((device float4*)(data0+alu0)) = float4((((val2.x-val0)*val1*val3.x)+val4.x),(((val2.y-val0)*val1*val3.y)+val4.y),(((val2.z-val0)*val1*val3.z)+val4.z),(((val2.w-val0)*val1*val3.w)+val4.w));
}
#include <metal_stdlib>
using namespace metal;
kernel void r_144_4_8_96_4(device float* data0, device float* data1, device float* data2, device float* data3, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int lidx0 = lid.x; /* 4 */
  threadgroup float temp0[128];
  int gidx0 = gid.x; /* 144 */
  int lidx1 = lid.y; /* 8 */
  int alu0 = ((gidx0<<4)+lidx0);
  int alu1 = (lidx0<<5);
  int alu2 = (alu0+4);
  int alu3 = (alu0+8);
  int alu4 = (alu0+12);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int ridx0 = 0; ridx0 < 96; ridx0++) {
    int alu5 = (alu0+(lidx1*2304)+(ridx0*18432));
    float val0 = *(data1+lidx1+(ridx0<<3));
    float val1 = *(data2+alu5+4);
    acc1 = (acc1+(val0*val1));
    float val2 = *(data2+alu5+8);
    acc2 = (acc2+(val0*val2));
    float val3 = *(data2+alu5+12);
    acc3 = (acc3+(val0*val3));
    float val4 = *(data2+alu5);
    acc0 = (acc0+(val0*val4));
  }
  *((threadgroup float4*)(temp0+alu1+(lidx1<<2))) = float4(acc0,acc1,acc2,acc3);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx1)!=1)) {
    float val5 = *(data3+alu2);
    float val6 = *(data3+alu3);
    float val7 = *(data3+alu4);
    float val8 = *(data3+alu0);
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      float4 val9 = *((threadgroup float4*)(temp0+alu1+(ridx1<<2)));
      acc4 = (acc4+val9.x);
      acc5 = (acc5+val9.y);
      acc6 = (acc6+val9.z);
      acc7 = (acc7+val9.w);
    }
    *(data0+alu2) = (acc5+val5);
    *(data0+alu3) = (acc6+val6);
    *(data0+alu4) = (acc7+val7);
    *(data0+alu0) = (acc4+val8);
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void E_24_32_2(device float* data0, device float* data1, constant int& start_pos, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 24 */
  int lidx0 = lid.x; /* 32 */
  int alu0 = (gidx0<<5);
  int alu1 = ((start_pos*768)+alu0+lidx0);
  int alu2 = (alu0+lidx0);
  float val0 = *(data1+alu2+768);
  *(data0+alu1) = val0;
  float val1 = *(data1+alu2+1536);
  *(data0+alu1+98304) = val1;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_28start_pos2B129_4_16_3_4(device float* data0, device float* data1, device float* data2, constant int& start_pos, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* UOp(UOps.ALU, dtypes.int, arg=BinaryOps.ADD, src=(
  UOp(UOps.DEFINE_VAR, dtypes.int, arg=('start_pos', 1, 128), src=()),
  UOp(UOps.CONST, dtypes.int, arg=1, src=()),)) */
  int lidx0 = lid.x; /* 4 */
  int alu0 = (lidx0*192);
  int alu1 = (gidx0+(lidx0*((start_pos*3)+3)));
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    int alu2 = (ridx0<<2);
    int alu3 = ((gidx0*768)+alu0+alu2);
    int alu4 = (alu0+alu2);
    float4 val0 = *((device float4*)(data1+alu4+64));
    float4 val1 = *((device float4*)(data1+alu4+128));
    float4 val2 = *((device float4*)(data1+alu4));
    float4 val3 = *((device float4*)(data2+alu3+64));
    acc1 = (acc1+(val0.x*val3.x)+(val0.y*val3.y)+(val0.z*val3.z)+(val0.w*val3.w));
    float4 val4 = *((device float4*)(data2+alu3+128));
    acc2 = (acc2+(val1.x*val4.x)+(val1.y*val4.y)+(val1.z*val4.z)+(val1.w*val4.w));
    float4 val5 = *((device float4*)(data2+alu3));
    acc0 = (acc0+(val2.x*val5.x)+(val2.y*val5.y)+(val2.z*val5.z)+(val2.w*val5.w));
  }
  *(data0+alu1) = (acc0*0.125f);
  *(data0+start_pos+alu1+1) = (acc1*0.125f);
  *(data0+(start_pos<<1)+alu1+2) = (acc2*0.125f);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_3_4_28start_pos2B129(device float* data0, device float* data1, constant int& start_pos, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 3 */
  int lidx0 = lid.x; /* 4 */
  int alu0 = (start_pos+1);
  float acc0 = -INFINITY;
  for (int ridx0 = 0; ridx0 < alu0; ridx0++) {
    float val0 = *(data1+(gidx0*((start_pos<<2)+4))+(lidx0*alu0)+ridx0);
    acc0 = ((acc0<val0)?val0:acc0);
  }
  *(data0+(gidx0<<2)+lidx0) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_3_4_28start_pos2B129n1(device float* data0, device float* data1, device float* data2, constant int& start_pos, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 3 */
  int lidx0 = lid.x; /* 4 */
  int alu0 = (start_pos+1);
  int alu1 = ((gidx0<<2)+lidx0);
  float val0 = *(data2+alu1);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < alu0; ridx0++) {
    float val1 = *(data1+(gidx0*((start_pos<<2)+4))+(lidx0*alu0)+ridx0);
    acc0 = (acc0+exp2(((val1-val0)*1.4426950408889634f)));
  }
  *(data0+alu1) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void E_28start_pos2B129_4_3(device float* data0, device float* data1, device float* data2, device float* data3, constant int& start_pos, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* UOp(UOps.ALU, dtypes.int, arg=BinaryOps.ADD, src=(
  UOp(UOps.DEFINE_VAR, dtypes.int, arg=('start_pos', 1, 128), src=()),
  UOp(UOps.CONST, dtypes.int, arg=1, src=()),)) */
  int lidx0 = lid.x; /* 4 */
  int alu0 = (lidx0*3);
  int alu1 = (alu0+1);
  int alu2 = (alu0+2);
  int alu3 = (gidx0+(lidx0*((start_pos*3)+3)));
  int alu4 = (start_pos+alu3+1);
  int alu5 = ((start_pos<<1)+alu3+2);
  float val0 = *(data1+alu3);
  float val1 = *(data1+alu4);
  float val2 = *(data1+alu5);
  float val3 = *(data2+alu1);
  float val4 = *(data2+alu2);
  float val5 = *(data2+alu0);
  float val6 = *(data3+alu1);
  *(data0+alu4) = (exp2(((val1-val3)*1.4426950408889634f))*(1/val6));
  float val7 = *(data3+alu2);
  *(data0+alu5) = (exp2(((val2-val4)*1.4426950408889634f))*(1/val7));
  float val8 = *(data3+alu0);
  *(data0+alu3) = (exp2(((val0-val5)*1.4426950408889634f))*(1/val8));
}
#include <metal_stdlib>
using namespace metal;
kernel void r_3_4_4_16_28start_pos2B129(device float* data0, device float* data1, device float* data2, constant int& start_pos, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 4 */
  int gidx1 = gid.y; /* 3 */
  int lidx0 = lid.x; /* 4 */
  int lidx1 = lid.y; /* 16 */
  int alu0 = (start_pos+1);
  int alu1 = ((gidx1<<8)+(gidx0<<4)+(lidx0<<6)+lidx1);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < alu0; ridx0++) {
    float val0 = *(data1+(gidx1*((start_pos<<2)+4))+(lidx0*alu0)+ridx0);
    float val1 = *(data2+alu1+(ridx0*768)+98304);
    acc0 = (acc0+(val0*val1));
  }
  *(data0+alu1) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_48_4_8_96_4(device float* data0, device float* data1, device float* data2, device float* data3, device float* data4, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int lidx0 = lid.x; /* 4 */
  threadgroup float temp0[128];
  int gidx0 = gid.x; /* 48 */
  int lidx1 = lid.y; /* 8 */
  int alu0 = ((gidx0<<4)+lidx0);
  int alu1 = (lidx0<<5);
  int alu2 = (alu0+4);
  int alu3 = (alu0+8);
  int alu4 = (alu0+12);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int ridx0 = 0; ridx0 < 96; ridx0++) {
    int alu5 = (alu0+(lidx1*768)+(ridx0*6144));
    float val0 = *(data2+lidx1+(ridx0<<3));
    float val1 = *(data3+alu5+4);
    acc1 = (acc1+(val0*val1));
    float val2 = *(data3+alu5+8);
    acc2 = (acc2+(val0*val2));
    float val3 = *(data3+alu5+12);
    acc3 = (acc3+(val0*val3));
    float val4 = *(data3+alu5);
    acc0 = (acc0+(val0*val4));
  }
  *((threadgroup float4*)(temp0+alu1+(lidx1<<2))) = float4(acc0,acc1,acc2,acc3);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx1)!=1)) {
    float val5 = *(data1+alu2);
    float val6 = *(data1+alu3);
    float val7 = *(data1+alu4);
    float val8 = *(data1+alu0);
    float val9 = *(data4+alu2);
    float val10 = *(data4+alu3);
    float val11 = *(data4+alu4);
    float val12 = *(data4+alu0);
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      float4 val13 = *((threadgroup float4*)(temp0+alu1+(ridx1<<2)));
      acc4 = (acc4+val13.x);
      acc5 = (acc5+val13.y);
      acc6 = (acc6+val13.z);
      acc7 = (acc7+val13.w);
    }
    *(data0+alu2) = (val5+acc5+val9);
    *(data0+alu3) = (val6+acc6+val10);
    *(data0+alu4) = (val7+acc7+val11);
    *(data0+alu0) = (val8+acc4+val12);
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void r_192_4_8_96_4(device float* data0, device float* data1, device float* data2, device float* data3, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int lidx0 = lid.x; /* 4 */
  threadgroup float temp0[128];
  int gidx0 = gid.x; /* 192 */
  int lidx1 = lid.y; /* 8 */
  int alu0 = ((gidx0<<4)+lidx0);
  int alu1 = (lidx0<<5);
  int alu2 = (alu0+4);
  int alu3 = (alu0+8);
  int alu4 = (alu0+12);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int ridx0 = 0; ridx0 < 96; ridx0++) {
    int alu5 = (alu0+(lidx1*3072)+(ridx0*24576));
    float val0 = *(data1+lidx1+(ridx0<<3));
    float val1 = *(data2+alu5+4);
    acc1 = (acc1+(val0*val1));
    float val2 = *(data2+alu5+8);
    acc2 = (acc2+(val0*val2));
    float val3 = *(data2+alu5+12);
    acc3 = (acc3+(val0*val3));
    float val4 = *(data2+alu5);
    acc0 = (acc0+(val0*val4));
  }
  *((threadgroup float4*)(temp0+alu1+(lidx1<<2))) = float4(acc0,acc1,acc2,acc3);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx1)!=1)) {
    float val5 = *(data3+alu2);
    float val6 = *(data3+alu3);
    float val7 = *(data3+alu4);
    float val8 = *(data3+alu0);
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      float4 val9 = *((threadgroup float4*)(temp0+alu1+(ridx1<<2)));
      acc4 = (acc4+val9.x);
      acc5 = (acc5+val9.y);
      acc6 = (acc6+val9.z);
      acc7 = (acc7+val9.w);
    }
    float alu6 = (acc4+val8);
    float alu7 = (acc5+val5);
    float alu8 = (acc6+val6);
    float alu9 = (acc7+val7);
    *(data0+alu2) = ((1/(exp2(((alu7+(alu7*alu7*alu7*0.044715f))*-2.302208198144325f))+1.0f))*alu7);
    *(data0+alu3) = ((1/(exp2(((alu8+(alu8*alu8*alu8*0.044715f))*-2.302208198144325f))+1.0f))*alu8);
    *(data0+alu4) = ((1/(exp2(((alu9+(alu9*alu9*alu9*0.044715f))*-2.302208198144325f))+1.0f))*alu9);
    *(data0+alu0) = ((1/(exp2(((alu6+(alu6*alu6*alu6*0.044715f))*-2.302208198144325f))+1.0f))*alu6);
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void r_48_4_8_384_4(device float* data0, device float* data1, device float* data2, device float* data3, device float* data4, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int lidx0 = lid.x; /* 4 */
  threadgroup float temp0[128];
  int gidx0 = gid.x; /* 48 */
  int lidx1 = lid.y; /* 8 */
  int alu0 = ((gidx0<<4)+lidx0);
  int alu1 = (lidx0<<5);
  int alu2 = (alu0+4);
  int alu3 = (alu0+8);
  int alu4 = (alu0+12);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int ridx0 = 0; ridx0 < 384; ridx0++) {
    int alu5 = (alu0+(lidx1*768)+(ridx0*6144));
    float val0 = *(data2+lidx1+(ridx0<<3));
    float val1 = *(data3+alu5+4);
    acc1 = (acc1+(val0*val1));
    float val2 = *(data3+alu5+8);
    acc2 = (acc2+(val0*val2));
    float val3 = *(data3+alu5+12);
    acc3 = (acc3+(val0*val3));
    float val4 = *(data3+alu5);
    acc0 = (acc0+(val0*val4));
  }
  *((threadgroup float4*)(temp0+alu1+(lidx1<<2))) = float4(acc0,acc1,acc2,acc3);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((bool)(lidx1)!=1)) {
    float val5 = *(data1+alu2);
    float val6 = *(data1+alu3);
    float val7 = *(data1+alu4);
    float val8 = *(data1+alu0);
    float val9 = *(data4+alu2);
    float val10 = *(data4+alu3);
    float val11 = *(data4+alu4);
    float val12 = *(data4+alu0);
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      float4 val13 = *((threadgroup float4*)(temp0+alu1+(ridx1<<2)));
      acc4 = (acc4+val13.x);
      acc5 = (acc5+val13.y);
      acc6 = (acc6+val13.z);
      acc7 = (acc7+val13.w);
    }
    *(data0+alu2) = (val5+acc5+val9);
    *(data0+alu3) = (val6+acc6+val10);
    *(data0+alu4) = (val7+acc7+val11);
    *(data0+alu0) = (val8+acc4+val12);
  }
}
#include <metal_stdlib>
using namespace metal;
kernel void E_n11(device unsigned int* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  unsigned int val0 = *(data0+0);
  *(data0+0) = (val0+1u);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_50257_192_4(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 50257 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 192; ridx0++) {
    int alu0 = (ridx0<<2);
    float4 val0 = *((device float4*)(data1+alu0));
    float4 val1 = *((device float4*)(data2+(gidx0*768)+alu0));
    acc0 = (acc0+(val0.x*val1.x)+(val0.y*val1.y)+(val0.z*val1.z)+(val0.w*val1.w));
  }
  *(data0+gidx0) = (acc0*1.25f);
}
#include <metal_stdlib>
using namespace metal;
kernel void r_29_1733n3(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 29 */
  float acc0 = -INFINITY;
  for (int ridx0 = 0; ridx0 < 1733; ridx0++) {
    float val0 = *(data1+(gidx0*1733)+ridx0);
    acc0 = ((acc0<val0)?val0:acc0);
  }
  *(data0+gidx0) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void r_29_1733n4(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 29 */
  float val0 = *(data2+0);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 1733; ridx0++) {
    float val1 = *(data1+(gidx0*1733)+ridx0);
    acc0 = (acc0+exp2(((val1-val0)*1.4426950408889634f)));
  }
  *(data0+gidx0) = acc0;
}
#include <metal_stdlib>
using namespace metal;
kernel void E_50257n1(device float* data0, device float* data1, device float* data2, device float* data3, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 50257 */
  float val0 = *(data1+gidx0);
  float val1 = *(data2+0);
  float val2 = *(data3+0);
  *(data0+gidx0) = (exp2(((val0-val1)*1.4426950408889634f))*(1/val2));
}

