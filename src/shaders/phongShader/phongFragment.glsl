#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 100
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight; //表示的是转换到light空间的坐标

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float Bias(){
  //解决shadow bias 因为shadow map的精度有限，当要渲染的fragment在light space中距Light很远的时候，就会有多个附近的fragement会samper shadow map中同一个texel,但是即使这些fragment在camera view space中的深度值z随xy变化是值变化是很大的，
  //但他们在light space 中的z值(shadow map中的值)却没变或变化很小，这是因为shadow map分辨率低，采样率低导致精度低，不能准确的记录这些细微的变化
 
  // calculate bias (based on depth map resolution and slope)  vec3 lightDir = normalize(uLightPos);
  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.005);
  return  bias;
}


float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
  float aveZ = 0.0;
  int blockNum = 0;
  float bias = Bias();

  poissonDiskSamples(uv);
  for(int i = 0;i<NUM_SAMPLES;i++) {
    float depthInShadowmap = unpack(texture2D(shadowMap,uv+poissonDisk[i]/float(2048) *50.0));
    if(zReceiver > depthInShadowmap + bias) {
      aveZ += depthInShadowmap;
      blockNum++;
    }
  }

  if(blockNum == 0) return -1.0;
  else return aveZ / float(blockNum);
}


float PCF(sampler2D shadowMap, vec4 coords) {
  poissonDiskSamples(coords.xy);
  
  float bias = Bias();
  float aveShadow = 0.0;

  float texture_size = 2048.0;
  float filter_stride = 5.0;
  float filter_area = filter_stride / texture_size;

  for(int i = 0;i<NUM_SAMPLES;i++) {

    float depthInShadowmap = unpack(texture2D(shadowMap,coords.xy+poissonDisk[i]*filter_area).rgba);
    aveShadow += ((depthInShadowmap + bias)< coords.z?0.0:1.0);
  }

  return aveShadow/float(NUM_SAMPLES);
}

float PCSS(sampler2D shadowMap, vec4 coords){

  // STEP 1: avgblocker depth
  float blockDepth = 0.0;
  float zReceiver = coords.z;
  blockDepth = findBlocker(shadowMap,coords.xy,zReceiver);

  // STEP 2: penumbra size
  if(blockDepth < EPS) return 1.0;
  else if(blockDepth > 1.0 + EPS) return 0.0;
  
  float w_light = 1.0;
  float w_penumbraSize = (zReceiver - blockDepth) * w_light / blockDepth;
  
  // STEP 3: filtering

  poissonDiskSamples(coords.xy);
  
  float bias = Bias();
  float aveShadow = 0.0;
  float texture_size = 2048.0;
  float filter_stride = 5.0;
  float filter_area = filter_stride / texture_size;

  for(int i = 0;i<NUM_SAMPLES;i++) {

    float depthInShadowmap = unpack(texture2D(shadowMap,coords.xy+poissonDisk[i]*filter_area * w_penumbraSize).rgba);
    aveShadow += ((depthInShadowmap + bias)< coords.z?0.0:1.0);
  }

  return aveShadow/float(NUM_SAMPLES);

}

float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){

  float bias = Bias();
  vec4 SM = texture2D(shadowMap,shadowCoord.xy);

  float depth = unpack(SM);

  if(depth > shadowCoord.z - bias) 
    return 1.0;
  else 
    return 0.0;
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {

  float visibility = 1.0;

  vec3 shadowCoord = vPositionFromLight.xyz / vPositionFromLight.w;
  shadowCoord = shadowCoord * 0.5 + 0.5;
  //visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  //visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));
    
  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  //gl_FragColor = vec4(phongColor, 1.0);
}