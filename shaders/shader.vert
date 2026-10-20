#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

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

layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularColor;
    float shininess;
    int useLighting;
    float _pad0;
    float _pad1;
} object;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) flat out vec4 fragAlbedo;
layout(location = 3) flat out vec4 fragSpecColor;
layout(location = 4) flat out float fragShininess;
layout(location = 5) flat out int fragUseLighting;

void main() {
    vec4 worldPos = object.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    fragNormal = normalize(mat3(object.normalMatrix) * inNormal);
    fragAlbedo = object.albedo;
    fragSpecColor = object.specularColor;
    fragShininess = object.shininess;
    fragUseLighting = object.useLighting;
    gl_Position = scene.viewProj * worldPos;
}