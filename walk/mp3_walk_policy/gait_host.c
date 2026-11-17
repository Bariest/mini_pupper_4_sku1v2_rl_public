/* Host wrapper around minipupperesp's own gait and IK (stanford_gait.c, stanford_kinematics.c, compiled
 * unchanged from the minipupperesp checkout) and its servo_write tick formula (main.c:2526), so that the host
 * produces exactly the ticks the robot firmware would send. Built by firmware_gait.py. */
#include "stanford_gait.h"
#include "stanford_kinematics.h"

void gh_reset(float height_mm) { stanford_gait_reset(height_mm); }

void gh_step(float vx, float vy, float wz, float height_mm, float clearance_mm, float *feet12)
{
    sg_foot_t f[4];
    stanford_gait_step(vx, vy, wz, height_mm, clearance_mm, f);
    for (int i = 0; i < 4; i++) { feet12[3*i] = f[i].x; feet12[3*i+1] = f[i].y; feet12[3*i+2] = f[i].z; }
}

/* servo_write(i, sdeg[i] + offset[i]) for i = 1..12, with scale_dd = 3100 (main.c:2526-2531) */
void gh_ticks(const float *feet12, const float *offset13, int *ticks13)
{
    sg_foot_t f[4];
    float sdeg[13];
    for (int i = 0; i < 4; i++) { f[i].x = feet12[3*i]; f[i].y = feet12[3*i+1]; f[i].z = feet12[3*i+2]; }
    stanford_kinematics_servo_deg(f, sdeg);
    ticks13[0] = 0;
    for (int i = 1; i <= 12; i++) {
        float ang = sdeg[i] + offset13[i];
        int sig = 511 + (int)(ang * 10240.0f / (float)3100);
        if (sig < 0) sig = 0;
        if (sig > 1023) sig = 1023;
        ticks13[i] = sig;
    }
}
