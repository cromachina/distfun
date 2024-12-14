#version 460
#include "noise"
#define PI 3.1415926538

uniform mat4 view_matrix;
uniform vec2 resolution;
uniform float epsilon;
uniform float fov;
uniform float time;
uniform int frame;

out vec4 frag_color;

vec2 grad(vec3 v, float eps)
{
    float vx = snoise(v + vec3(eps, 0, 0));
    float vy = snoise(v + vec3(0, eps, 0));
    float nv = snoise(v);
    return vec2(nv) - vec2(vx, vy);
}

vec3 grad(vec4 v, float eps)
{
    float vx = snoise(v + vec4(eps, 0, 0, 0));
    float vy = snoise(v + vec4(0, eps, 0, 0));
    float vz = snoise(v + vec4(0, 0, eps, 0));
    float nv = snoise(v);
    return vec3(nv) - vec3(vx, vy, vz);
}

vec3 warp(vec3 p, int octaves, float scale)
{
    for (int i = 1; i <= octaves; i++)
    {
        p += grad(vec4(p / i, time * 0.1), 1) * scale;
    }
    return p;
}

vec3 op_repeat(vec3 p, vec3 c)
{
    return mod(p, c) - 0.5 * c;
}

float op_subtract(float d1, float d2)
{
    return max(d1, -d2);
}

float box(vec3 p, vec3 bounds)
{
    return length(max(abs(p) - bounds, vec3(0)));
}

float sphere(vec3 p, float radius)
{
    return length(p) - radius;
}

float plane(vec3 p, vec4 n)
{
    return dot(p, n.xyz) + n.w;
}

float soft_min(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0 );
    return mix(b, a, h) - k * h * (1.0 - h);
}

float cylinder(vec3 p, vec3 c)
{
    return length(p.xz - c.xy) - c.z;
}

float capped_cylinder(vec3 p, vec2 h)
{
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float bamboo_section(vec3 p)
{
    float shaft = capped_cylinder(p, vec2(0.1, 1.0));
    float ridge = capped_cylinder(p, vec2(0.12, 0.005));
    float ridge_cut = capped_cylinder(p, vec2(1, 0.006));
    float ridge_inset = capped_cylinder(p, vec2(0.119, 0.01));
    return min(op_subtract(soft_min(shaft, ridge, 0.05),
                   ridge_cut),
               ridge_inset);
}

float bamboo_shaft(vec3 p)
{
    return bamboo_section(op_repeat(p, vec3(5, 1, 5)));
}

float map(vec3 p)
{
    float res;
    res = plane(p, vec4(0, 1, 0, 1));
    res = min(res, box(p, vec3(1)));
    res = min(res, sphere(p - vec3(1), 0.5));
    res = min(res, bamboo_shaft(p));
    return res;
}

vec3 normal_map(vec3 p, float d, float delta)
{
    vec2 eps = vec2(delta, 0);
    float dx = map(p + eps.xyy) - d;
    float dy = map(p + eps.yxy) - d;
    float dz = map(p + eps.yyx) - d;
    return normalize(vec3(dx, dy, dz));
}

vec3 normal_light(vec3 p, float radius, float d, float delta)
{
    vec2 eps = vec2(delta, 0);
    float dx = sphere(p + eps.xyy, radius) - d;
    float dy = sphere(p + eps.yxy, radius) - d;
    float dz = sphere(p + eps.yyx, radius) - d;
    return normalize(vec3(dx, dy, dz));
}

float ambient_occlusion(vec3 p, vec3 normal, int steps, float delta)
{
    float res = 0;
    float total = 0;

    for (int i = 1; i <= steps; ++i)
    {
        vec3 q = p + normal * (delta * i);
        float d = map(q);
        float lpq = length(p - q);

        total += lpq;
        res += lpq - d;
    }

    return 1 - res / total;
}

float hard_shadow(vec3 rayOrig, vec3 rayDir, float tmin, float tmax)
{
    for (float t = tmin; t < tmax;)
    {
        float h = map(rayOrig + rayDir * t);

        if (h < epsilon)
        {
            return 0.0;
        }

        t += h;
    }

    return 1.0;
}

float soft_shadow(vec3 rayOrig, vec3 rayDir, float tmin, float tmax, float k)
{
    float res = 1.0;

    for (float t = tmin; t < tmax;)
    {
        float h = map(rayOrig + rayDir * t);

        if (h < epsilon)
        {
            return 0.0;
        }

        res = min(res, k * h / t);
        t += h;
    }

    return res;
}

vec4 quat_mult(vec4 q1, vec4 q2)
{
    return vec4(
    q1.w * q2.x + q1.x * q2.w + q1.z * q2.y - q1.y * q2.z,
    q1.w * q2.y + q1.y * q2.w + q1.x * q2.z - q1.z * q2.x,
    q1.w * q2.z + q1.z * q2.w + q1.y * q2.x - q1.x * q2.y,
    q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

vec4 quat(vec3 axis, float angle)
{
    float s = sin(angle / 2);
    return vec4(axis * s, cos(angle / 2));
}

vec3 quat_rot(vec4 q, vec3 v)
{
    vec3 temp = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, temp);
}

vec3 fisheye(vec2 screen, float fov)
{
    float u = screen.x;
    float v = screen.y;

    float phi = atan(v, u) + 2.0 * PI / 2.0;
    float theta = fov * sqrt(u * u + v * v);

    float xt = -sin(theta) * cos(phi);
    float yt = -sin(theta) * sin(phi);
    float zt = cos(theta);

    return vec3(xt, yt, zt);
}

void main()
{
    /*if ((int(gl_FragCoord.y + gl_FragCoord.x) + frame) % 4 == 0)
    {
        discard;
    }
    */

    vec3 eye_pos = vec3(view_matrix * vec4(0, 0, 0, 1));
    vec3 eye_right = normalize(eye_pos - vec3(view_matrix * vec4(1, 0, 0, 1)));
    vec3 eye_up = normalize(eye_pos - vec3(view_matrix * vec4(0, 1, 0, 1)));
    vec3 eye_forward = normalize(eye_pos - vec3(view_matrix * vec4(0, 0, 1, 1)));

    // Normalized screen coordinates
    float sx = gl_FragCoord.x * 2.0 / resolution.y - (resolution.x / resolution.y);
    float sy = gl_FragCoord.y * 2.0 / resolution.y - 1.0;

    vec2 ss = vec2(sx, -sy);

    // vec2 p = gl_FragCoord.xy / 200.0;
    // for (int i = 2; i <= 4; i++)
    // {
    //     p += grad(vec3(p / i, time * 0.1), 1);
    // }
    //
    //ss += p * 0.05;

    // Linear
    // float focal = 2 / tan(fov / 180 * PI);
    // vec3 ray_dir = normalize(eye_forward * focal + eye_right * u + eye_up * v);

    // Fisheye
    float fov_r = radians(fov) / 2;
    vec3 ray_dir = fisheye(ss, fov_r);
    ray_dir = normalize(eye_pos - vec3(view_matrix * vec4(ray_dir, 1)));

    // Light parameters
    vec3 light_pos = vec3(2, 4, 3);//vec3(0, 0.1 + sin(time / 1000.0), 1);
    float light_size = .4;
    float light_power = 10;
    vec4 light_color = vec4(1, 0.9, .5, 1);

    vec4 fog_color = vec4(0, 0, 0.1, 1);
    float fog_dist = 50;
    vec4 ambient_light = fog_color + 0.01;

    // Final color
    vec4 color = fog_color;

    // Object color
    const vec4 base_color = vec4(0.9, 0.9, 1, 1);

    float t = 0.0;

    int max_steps = 100;

    for (int i = 0; i < max_steps; ++i)
    {
        vec3 p = eye_pos + ray_dir * t;
        p = warp(p, 2, 0.05);

        float light_d = sphere(p - light_pos, light_size);
        float sdf_dist = min(map(p), light_d);

        if (sdf_dist < epsilon)
        {
            vec3 normal = normal_map(p, sdf_dist, epsilon);
            vec3 light_ray = light_pos - p;
            vec3 light_dir = normalize(light_ray);
            float light_dist = length(light_ray);
            float light_intensity = dot(normal, light_dir);
            float attenuation = pow(light_dist, 2);
            light_intensity /= attenuation;
            light_intensity *= light_power;

            if (light_dist <= light_size + 0.01)
            {
                // Color the light object.
                color = pow(1 - dot(normal_light(p - light_pos, light_size, sdf_dist, 0.001), -ray_dir), 0.9) * 2 * light_color * light_power;
            }
            else
            {
                color = base_color * light_intensity * light_color;
                vec4 ambient_surface = base_color * ambient_light;
                vec4 lit_surface = ambient_surface + (light_intensity * light_color);

                // Darken by shadow.
                //float shadowBias = mix(0.01, 0.1, 1.0 - abs(dot(normal, ray_dir)) * abs(dot(normal, light_dir)));
                float shadow = soft_shadow(p, light_dir, 0.01, light_dist, 128);
                color = mix(ambient_surface, lit_surface, shadow);

                // Darken by ambient occlusion.
                //float ambient_occ = ambient_occlusion(p, normal, 5, 0.05);
                //color = mix(color * vec4(ambient_min), color, ambient_occ);
                //color = vec4(ambient_occ);

                // Color by normal.
                //color = vec4((normal + 1) / 2, 1);

                // Color by steps.
                color = vec4(1.0 - i / float(max_steps));
            }

            // Add fog.
            float p_dist = length(p - eye_pos);
            color = mix(color, fog_color, 1 - exp(-p_dist / fog_dist));

            break;
        }

        t += sdf_dist;
    }

    frag_color = color;
}
