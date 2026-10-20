#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) flat in vec4 fragAlbedo;
layout(location = 3) flat in vec4 fragSpecColor;
layout(location = 4) flat in float fragShininess;
layout(location = 5) flat in int fragUseLighting;

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 angles;
};

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    vec3 cameraPos;
    float ambientStrength;
    vec3 ambientColor;
    float _pad1;
    vec3 dirLightDirection;
    float _pad2;
    vec3 dirLightColor;
    float dirLightIntensity;
    int numPointLights;
    int numSpotLights;
    int showLightGizmos;
    float _pad3;
} scene;

layout(set = 0, binding = 1) readonly buffer PointLightSSBO {
    PointLight pointLights[];
};

layout(set = 0, binding = 2) readonly buffer SpotLightSSBO {
    SpotLight spotLights[];
};

layout(location = 0) out vec4 outColor;

void main() {
    if (fragUseLighting == 0) {
        outColor = vec4(fragAlbedo.rgb, 1.0);
        return;
    }

    vec3 N = normalize(fragNormal);
    vec3 viewDir = normalize(scene.cameraPos - fragWorldPos);
    vec3 albedo = fragAlbedo.rgb;
    vec3 specColor = fragSpecColor.rgb;

    vec3 ambient = scene.ambientStrength * scene.ambientColor * albedo;

    vec3 result = ambient;

    vec3 lightDir = normalize(-scene.dirLightDirection);
    float diff = max(dot(N, lightDir), 0.0);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(N, halfDir), 0.0), fragShininess);
    result += (diff * albedo + spec * specColor) * scene.dirLightColor * scene.dirLightIntensity;

    for (int i = 0; i < scene.numPointLights; ++i) {
        vec3 lightVec = pointLights[i].position.xyz - fragWorldPos;
        float dist = length(lightVec);
        float attenuation = 1.0 / (dist * dist + 0.0001);
        vec3 pLightDir = lightVec / dist;

        float pDiff = max(dot(N, pLightDir), 0.0);
        vec3 pHalf = normalize(pLightDir + viewDir);
        float pSpec = pow(max(dot(N, pHalf), 0.0), fragShininess);

        vec3 pColor = pointLights[i].color.rgb * pointLights[i].color.w;
        result += (pDiff * albedo + pSpec * specColor) * pColor * attenuation;
    }

    for (int i = 0; i < scene.numSpotLights; ++i) {
        vec3 lightVec = spotLights[i].position.xyz - fragWorldPos;
        float dist = length(lightVec);
        float attenuation = 1.0 / (dist * dist + 0.0001);
        vec3 sLightDir = lightVec / dist;

        float sDiff = max(dot(N, sLightDir), 0.0);
        vec3 sHalf = normalize(sLightDir + viewDir);
        float sSpec = pow(max(dot(N, sHalf), 0.0), fragShininess);

        float theta = dot(normalize(fragWorldPos - spotLights[i].position.xyz),
                          normalize(spotLights[i].direction.xyz));
        float innerCos = spotLights[i].angles.x;
        float outerCos = spotLights[i].angles.y;
        float spotFactor = clamp((theta - outerCos) / (innerCos - outerCos + 0.0001), 0.0, 1.0);

        vec3 sColor = spotLights[i].color.rgb * spotLights[i].color.w;
        result += (sDiff * albedo + sSpec * specColor) * sColor * attenuation * spotFactor;
    }

    result = clamp(result, 0.0, 1.0);
    outColor = vec4(result, 1.0);
}