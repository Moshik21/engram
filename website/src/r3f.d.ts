/* eslint-disable @typescript-eslint/no-namespace */
import type { Object3DNode, MaterialNode, BufferGeometryNode } from "@react-three/fiber";
import * as THREE from "three";

declare global {
  namespace JSX {
    interface IntrinsicElements {
      instancedMesh: Object3DNode<THREE.InstancedMesh, typeof THREE.InstancedMesh>;
      sphereGeometry: BufferGeometryNode<THREE.SphereGeometry, typeof THREE.SphereGeometry>;
      meshStandardMaterial: MaterialNode<THREE.MeshStandardMaterial, typeof THREE.MeshStandardMaterial>;
      meshBasicMaterial: MaterialNode<THREE.MeshBasicMaterial, typeof THREE.MeshBasicMaterial>;
      lineSegments: Object3DNode<THREE.LineSegments, typeof THREE.LineSegments>;
      bufferGeometry: BufferGeometryNode<THREE.BufferGeometry, typeof THREE.BufferGeometry>;
      bufferAttribute: any;
      lineBasicMaterial: MaterialNode<THREE.LineBasicMaterial, typeof THREE.LineBasicMaterial>;
      points: Object3DNode<THREE.Points, typeof THREE.Points>;
      pointsMaterial: MaterialNode<THREE.PointsMaterial, typeof THREE.PointsMaterial>;
      ambientLight: Object3DNode<THREE.AmbientLight, typeof THREE.AmbientLight>;
      pointLight: Object3DNode<THREE.PointLight, typeof THREE.PointLight>;
    }
  }
}
