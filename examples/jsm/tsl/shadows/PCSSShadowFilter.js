import { NearestFilter } from 'three/webgpu';
import { Fn, Loop, float, vec2, vec3, vec4, abs, cross, dot, fract, mix, normalize, max, step, tan, radians, uniform, reference, renderGroup, texture, cubeTexture, screenCoordinate, interleavedGradientNoise, vogelDiskSample, orthographicDepthToViewZ, perspectiveDepthToViewZ, logarithmicDepthToViewZ, viewZToPerspectiveDepth, viewZToReversedPerspectiveDepth, viewZToLogarithmicDepth } from 'three/tsl';

/**
 * Creates a shadow filter that implements Percentage-Closer Soft Shadows (PCSS).
 * Shadows are sharp where an occluder touches the receiver and get softer the
 * farther the receiver is from the occluder, like shadows cast by a light of
 * a certain size. Assign the result to `light.shadow.filterNode`.
 *
 * The filter works in three steps:
 * 1. Blocker search: the depths of the occluders around the receiver are averaged.
 * 2. Penumbra estimation: the size of the penumbra follows from the light size and
 * the distances of the receiver and the occluders to the light.
 * 3. Filtering: percentage-closer filtering with a kernel scaled to the penumbra size.
 *
 * Directional, spot and point lights are supported. The meaning of `lightSize` depends
 * on the light type:
 * - Directional lights: The tangent of the light's angular radius (e.g. `0.0047` for the sun).
 * - Spot and point lights: The radius of the light in world units.
 *
 * ```js
 * const filter = PCSSShadowFilter( { lightSize: 0.2 } );
 *
 * light.shadow.filterNode = filter;
 *
 * filter.lightSize.value = 0.4; // can be changed at runtime
 * ```
 *
 * By default, the blocker search looks for occluders anywhere between the receiver and the
 * near plane of the shadow camera. When the shadow camera is far away from the scene, like
 * with cascaded shadow maps where each cascade places its own shadow camera, `maxOccluderDistance`
 * limits the search to occluders up to that distance from the receiver (in world units). This
 * reduces noise and keeps penumbrae consistent between cascades. Use at least the largest
 * distance between an occluder and the surfaces it shadows, since the penumbrae of occluders
 * farther away are cut off.
 *
 * @tsl
 * @function
 * @param {Object} [parameters] - The configuration parameters.
 * @param {number|Node<float>} [parameters.lightSize=0.1] - The size of the light.
 * @param {?(number|Node<float>)} [parameters.maxOccluderDistance=null] - The maximum distance between a receiver and its occluders in world units. `null` means no limit.
 * @param {number} [parameters.blockerSearchSamples=32] - The number of samples of the blocker search.
 * @param {number} [parameters.filterSamples=32] - The number of samples of the filtering.
 * @return {Function} The shadow filter. Its `lightSize` and `maxOccluderDistance` properties hold the respective nodes.
 */
export const PCSSShadowFilter = ( { lightSize = 0.1, maxOccluderDistance = null, blockerSearchSamples = 32, filterSamples = 32 } = {} ) => {

	const lightSizeNode = ( lightSize.isNode === true ) ? lightSize : uniform( lightSize );
	const maxOccluderDistanceNode = ( maxOccluderDistance === null || maxOccluderDistance.isNode === true ) ? maxOccluderDistance : uniform( maxOccluderDistance );

	const filter = Fn( ( inputs, builder ) => {

		// PCSS reads raw depth values, so the shadow map must not use a comparison sampler
		// or linear filtering (an unfilterable depth texture is incomplete in WebGL 2)

		const { depthTexture } = inputs;

		depthTexture.compareFunction = null;
		depthTexture.minFilter = NearestFilter;
		depthTexture.magFilter = NearestFilter;

		// point lights pass the direction from the light to the fragment instead of shadow coordinates

		const params = { lightSize: lightSizeNode, maxOccluderDistance: maxOccluderDistanceNode, blockerSearchSamples, filterSamples };

		return ( inputs.bd3D !== undefined ) ? pointFilter( inputs, params, builder ) : planarFilter( inputs, params, builder );

	} );

	filter.lightSize = lightSizeNode;
	filter.maxOccluderDistance = maxOccluderDistanceNode;

	return filter;

};

// rotates the sampling pattern per pixel to trade banding for noise

const getSamplingAngle = () => interleavedGradientNoise( screenCoordinate.xy ).mul( Math.PI * 2 );

// directional and spot lights

function planarFilter( { depthTexture, shadowCoord, shadow, depthLayer }, { lightSize, maxOccluderDistance, blockerSearchSamples, filterSamples }, builder ) {

	const { reversedDepthBuffer, logarithmicDepthBuffer } = builder.renderer;
	const camera = shadow.camera;

	const mapSize = reference( 'mapSize', 'vec2', shadow ).setGroup( renderGroup );
	const near = reference( 'near', 'float', camera ).setGroup( renderGroup );
	const far = reference( 'far', 'float', camera ).setGroup( renderGroup );
	const zoom = reference( 'zoom', 'float', camera ).setGroup( renderGroup );

	const receiverDepth = shadowCoord.z;

	let toDistance, receiverDistance, searchRadius, getFilterRadius;

	if ( camera.isOrthographicCamera === true ) {

		// world space size of the area covered by the shadow map

		const left = reference( 'left', 'float', camera ).setGroup( renderGroup );
		const right = reference( 'right', 'float', camera ).setGroup( renderGroup );
		const top = reference( 'top', 'float', camera ).setGroup( renderGroup );
		const bottom = reference( 'bottom', 'float', camera ).setGroup( renderGroup );

		const frustumSize = vec2( right.sub( left ), top.sub( bottom ) ).div( zoom );

		toDistance = ( depth ) => orthographicDepthToViewZ( depth, near, far ).negate();

		receiverDistance = toDistance( receiverDepth );

		// the light's rays are spread by its angular size, so occluders up to the near plane can block the receiver

		let occluderDistance = receiverDistance.sub( near );

		if ( maxOccluderDistance !== null ) occluderDistance = occluderDistance.min( maxOccluderDistance );

		searchRadius = occluderDistance.mul( lightSize ).div( frustumSize );
		getFilterRadius = ( blockerDistance ) => receiverDistance.sub( blockerDistance ).mul( lightSize ).div( frustumSize );

	} else {

		// size of the area covered by the shadow map at a distance of one unit

		const fov = reference( 'fov', 'float', camera ).setGroup( renderGroup );
		const aspect = reference( 'aspect', 'float', camera ).setGroup( renderGroup );

		const tanHalfFov = tan( radians( fov ).mul( 0.5 ) ).div( zoom );
		const frustumScale = vec2( tanHalfFov.mul( aspect ), tanHalfFov ).mul( 2 );

		toDistance = ( logarithmicDepthBuffer === true )
			? ( depth ) => logarithmicDepthToViewZ( depth, near, far ).negate()
			: ( depth ) => perspectiveDepthToViewZ( depth, near, far ).negate();

		receiverDistance = toDistance( receiverDepth );

		// similar triangles: the cone between the receiver and the light, measured on the near plane
		// or closer to the receiver if the occluder distance is limited

		let searchDistance = near;

		if ( maxOccluderDistance !== null ) searchDistance = receiverDistance.sub( maxOccluderDistance ).max( near );

		searchRadius = lightSize.mul( receiverDistance.sub( searchDistance ) ).div( receiverDistance.mul( searchDistance ) ).div( frustumScale );
		getFilterRadius = ( blockerDistance ) => lightSize.mul( receiverDistance.sub( blockerDistance ) ).div( blockerDistance.mul( receiverDistance ) ).div( frustumScale );

	}

	// each sample gathers the four texels around it, like hardware PCF does

	const gatherDepths = ( uv ) => {

		let depths = texture( depthTexture, uv ).gather();

		if ( depthTexture.isArrayTexture ) {

			depths = depths.depth( depthLayer );

		}

		return depths.toConst();

	};

	const getLit = ( depths ) => reversedDepthBuffer ? step( depths, vec4( receiverDepth ) ) : step( vec4( receiverDepth ), depths );

	const phi = getSamplingAngle();

	// 1. blocker search

	const blockerDepthSum = float( 0 ).toVar();
	const blockerCount = float( 0 ).toVar();

	Loop( blockerSearchSamples, ( { i } ) => {

		const depths = gatherDepths( shadowCoord.xy.add( vogelDiskSample( i, blockerSearchSamples, phi ).mul( searchRadius ) ) );
		const blockers = vec4( 1 ).sub( getLit( depths ) ).toConst();

		blockerDepthSum.addAssign( dot( blockers, depths ) );
		blockerCount.addAssign( dot( blockers, vec4( 1 ) ) );

	} );

	// 2. penumbra estimation

	// if the search found no blockers, a sparse search may have missed them, so fall back to a hard shadow test

	const blockerDistance = blockerCount.greaterThan( 0 ).select( toDistance( blockerDepthSum.div( blockerCount.max( 1 ) ) ), receiverDistance );
	const filterRadius = getFilterRadius( blockerDistance ).max( vec2( 1 ).div( mapSize ) ).toConst();

	// 3. filtering

	const lit = float( 0 ).toVar();

	Loop( filterSamples, ( { i } ) => {

		const uv = shadowCoord.xy.add( vogelDiskSample( i, filterSamples, phi ).mul( filterRadius ) ).toConst();
		const texelLit = getLit( gatherDepths( uv ) ).toConst();

		// gathered texels are ordered ( 0, 1 ), ( 1, 1 ), ( 1, 0 ), ( 0, 0 ) relative to the texel below and left of uv

		const weight = fract( uv.mul( mapSize ).sub( 0.5 ) ).toConst();

		lit.addAssign( mix( mix( texelLit.w, texelLit.z, weight.x ), mix( texelLit.x, texelLit.y, weight.x ), weight.y ) );

	} );

	return lit.div( filterSamples );

}

// point lights

function pointFilter( { depthTexture, bd3D, distance, shadow }, { lightSize, maxOccluderDistance, blockerSearchSamples, filterSamples }, builder ) {

	const { reversedDepthBuffer, logarithmicDepthBuffer } = builder.renderer;

	const mapSize = reference( 'mapSize', 'vec2', shadow ).setGroup( renderGroup );
	const near = reference( 'near', 'float', shadow.camera ).setGroup( renderGroup );
	const far = reference( 'far', 'float', shadow.camera ).setGroup( renderGroup );
	const bias = reference( 'bias', 'float', shadow ).setGroup( renderGroup );

	// each cube face stores the depth along its axis, which is the distance along a direction times the direction's largest component

	let toFaceDepth, toFaceDistance;

	if ( logarithmicDepthBuffer === true ) {

		toFaceDepth = ( faceDistance ) => viewZToLogarithmicDepth( faceDistance.negate(), near, far );
		toFaceDistance = ( depth ) => logarithmicDepthToViewZ( depth, near, far ).negate();

	} else {

		toFaceDepth = ( faceDistance ) => reversedDepthBuffer ? viewZToReversedPerspectiveDepth( faceDistance.negate(), near, far ) : viewZToPerspectiveDepth( faceDistance.negate(), near, far );
		toFaceDistance = ( depth ) => perspectiveDepthToViewZ( depth, near, far ).negate();

	}

	// tangent space around the direction from the light to the fragment, offsets are angles in radians

	const absDir = abs( bd3D );
	const tangent = normalize( cross( bd3D, absDir.x.greaterThan( absDir.z ).select( vec3( 0, 1, 0 ), vec3( 1, 0, 0 ) ) ) ).toConst();
	const bitangent = cross( bd3D, tangent ).toConst();

	const sampleShadowMap = ( offset ) => {

		const direction = normalize( bd3D.add( tangent.mul( offset.x ).add( bitangent.mul( offset.y ) ) ) ).toConst();
		const absDirection = abs( direction );
		const maxAbs = max( absDirection.x, absDirection.y, absDirection.z ).toConst();

		const depth = cubeTexture( depthTexture, direction ).level( 0 ).toConst();
		const receiverDepth = toFaceDepth( distance.mul( maxAbs ) );

		const isBlocker = reversedDepthBuffer ? depth.greaterThan( receiverDepth.sub( bias ) ) : depth.lessThan( receiverDepth.add( bias ) );

		return { isBlocker, blockerDistance: toFaceDistance( depth ).div( maxAbs ) };

	};

	const phi = getSamplingAngle();

	// 1. blocker search: the cone between the receiver and the light, measured on the near plane
	// or closer to the receiver if the occluder distance is limited

	let searchDistance = near;

	if ( maxOccluderDistance !== null ) searchDistance = distance.sub( maxOccluderDistance ).max( near );

	const searchRadius = lightSize.mul( distance.sub( searchDistance ) ).div( distance.mul( searchDistance ) ).toConst();

	const blockerDistanceSum = float( 0 ).toVar();
	const blockerCount = float( 0 ).toVar();

	Loop( blockerSearchSamples, ( { i } ) => {

		const { isBlocker, blockerDistance } = sampleShadowMap( vogelDiskSample( i, blockerSearchSamples, phi ).mul( searchRadius ) );

		blockerDistanceSum.addAssign( isBlocker.select( blockerDistance, 0 ) );
		blockerCount.addAssign( isBlocker.select( 1, 0 ) );

	} );

	// 2. penumbra estimation

	// if the search found no blockers, a sparse search may have missed them, so fall back to a hard shadow test

	const blockerDistance = blockerCount.greaterThan( 0 ).select( blockerDistanceSum.div( blockerCount.max( 1 ) ), distance );
	const filterRadius = lightSize.mul( distance.sub( blockerDistance ) ).div( blockerDistance.mul( distance ) ).max( float( 2 ).div( mapSize.x ) ).toConst();

	// 3. filtering

	const lit = float( 0 ).toVar();

	Loop( filterSamples, ( { i } ) => {

		const { isBlocker } = sampleShadowMap( vogelDiskSample( i, filterSamples, phi ).mul( filterRadius ) );

		lit.addAssign( isBlocker.select( 0, 1 ) );

	} );

	return lit.div( filterSamples );

}
