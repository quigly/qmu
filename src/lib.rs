use std::hash::{Hash, Hasher};

use rand::{SeedableRng, Rng};

pub fn lerp(v0: f64, v1: f64, t: f64) -> f64
{
	(1.0 - t) * v0 + t * v1
}

pub fn bilerp(c00: f64, c10: f64, c01: f64, c11: f64, tx: f64, ty: f64) -> f64
{
	lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Camera
{
	pub transform: Transform,
	pub fov: f32,
	pub aspect: f32,
	pub z_near: f32,
	pub z_far: f32
}

impl Camera
{
	pub fn new(fov: f32, aspect: f32, z_near: f32, z_far: f32) -> Self
	{
		let transform = Transform::new();

		Camera
		{
			transform,
			fov,
			aspect,
			z_near,
			z_far
		}
	}

	pub fn move_direction(&mut self, direction: &Vector3, amount: f32)
	{
		self.transform.translation.x += direction.x * amount;
		self.transform.translation.y += direction.y * amount;
		self.transform.translation.z += direction.z * amount;
	}

	pub fn view_projection_matrix(&self) -> Matrix4
	{
		let camera_projection =  Matrix4::init_perspective(self.fov, self.aspect, self.z_near, self.z_far);
		let camera_rotation = self.transform.rotation.conjugate().to_rotation_matrix();
		let camera_position = self.transform.translation.mulf(-1.0);
		let camera_translation = Matrix4::init_translation(camera_position.x, camera_position.y, camera_position.z);

		camera_projection.mul(&camera_rotation.mul(&camera_translation))
	}

	pub fn projection_matrix(&self) -> Matrix4
	{
		Matrix4::init_perspective(self.fov, self.aspect, self.z_near, self.z_far)
	}

	pub fn view_matrix(&self) -> Matrix4
	{
		let camera_rotation = self.transform.rotation.conjugate().to_rotation_matrix();
		let camera_position = self.transform.translation.mulf(-1.0);
		let camera_translation = Matrix4::init_translation(camera_position.x, camera_position.y, camera_position.z);

		camera_rotation.mul(&camera_translation)
	}

	/*pub fn calculate_frustum(&self) -> Frustum
	{
		let projection_matrix: Matrix4 = self.projection_matrix().transpose();
		
		// Extract the column vectors of the projection matrix
		let col0 = projection_matrix.column(0).xyz();
		let col1 = projection_matrix.column(1).xyz();
		let col2 = projection_matrix.column(2).xyz();
		let col3 = projection_matrix.column(3).xyz();

		// Compute the equations of the left, right, top, bottom, near, and far planes
		let left = Plane::from_point_normal(Vector3::zero(), col3 + col0); // Left plane
		let right = Plane::from_point_normal(Vector3::zero(), col3 - col0); // Right plane
		let top = Plane::from_point_normal(Vector3::zero(), col3 + col1); // Top plane
		let bottom = Plane::from_point_normal(Vector3::zero(), col3 - col1); // Bottom plane
		let near = Plane::from_point_normal(Vector3::zero(), col3 + col2); // Near plane
		let far = Plane::from_point_normal(Vector3::zero(), col3 - col2); // Far plane

		Frustum { left, right, bottom, top, near, far }
	}*/
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Vector2
{
	pub x: f32,
	pub y: f32
}

impl Vector2
{
	pub fn new(x: f32, y: f32) -> Self
	{
		Self { x, y }
	}

	pub fn zero() -> Self
	{
		Self
		{
			x: 0.0,
			y: 0.0
		}
	}

	pub fn length(&self) -> f32
	{
		return (self.x * self.x + self.y * self.y).sqrt();
	}

	pub fn normalized(&self) -> Self
	{
		let length = self.length();
		return Vector2
		{
			x: self.x / length,
			y: self.y / length
		}
	}

	pub fn add(&self, r: &Vector2) -> Self
	{
		Self
		{
			x: self.x + r.x,
			y: self.y + r.y
		}
	}

	pub fn sub(&self, r: &Vector2) -> Self
	{
		Self
		{
			x: self.x - r.x,
			y: self.y - r.y
		}
	}

	pub fn mul(&self, r: &Vector2) -> Self
	{
		Self
		{
			x: self.x * r.x,
			y: self.y * r.y
		}
	}

	pub fn div(&self, r: &Vector2) -> Self
	{
		Self
		{
			x: self.x / r.x,
			y: self.y / r.y
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vector2i
{
	pub x: i32,
	pub y: i32
}

impl Vector2i
{
	pub fn new(x: i32, y: i32) -> Self
	{
		Self { x, y }
	}

	pub fn add(&self, r: &Self) -> Self
	{
		Self
		{
			x: self.x + r.x,
			y: self.y + r.y
		}
	}

	pub fn sub(&self, r: &Self) -> Self
	{
		Self
		{
			x: self.x - r.x,
			y: self.y - r.y
		}
	}

	pub fn mul(&self, r: &Self) -> Self
	{
		Self
		{
			x: self.x * r.x,
			y: self.y * r.y
		}
	}

	pub fn div(&self, r: &Self) -> Self
	{
		Self
		{
			x: self.x / r.x,
			y: self.y / r.y
		}
	}

	pub fn hash(&self) -> u64
	{
		let mut hasher = std::collections::hash_map::DefaultHasher::new();
		self.x.hash(&mut hasher);
		self.y.hash(&mut hasher);
		hasher.finish()
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Vector3
{
	pub x: f32,
	pub y: f32,
	pub z: f32
}

impl Vector3
{
	pub fn new(x: f32, y: f32, z: f32) -> Self
	{
		Self { x, y, z }
	}

	pub fn zero() -> Self
	{
		Self
		{
			x: 0.0,
			y: 0.0,
			z: 0.0
		}
	}

	pub fn length(&self) -> f32
	{
		return (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
	}

	pub fn max(&self) -> f32
	{
		self.x.max(self.y.max(self.z))
	}

	pub fn normalized(&self) -> Self
	{
		let length: f32 = self.length();

		return Vector3::new(self.x / length, self.y / length, self.z / length);
	}

	pub fn dot(&self, r: &Vector3) -> f32
	{
		return self.x * r.x + self.y * r.y + self.z * r.z;
	}

	pub fn cross(&self, r: &Vector3) -> Self
	{
		let x: f32 = self.y * r.z - self.z * r.y;
		let y: f32 = self.z * r.x - self.x * r.z;
		let z: f32 = self.x * r.y - self.y * r.x;
		
		return Vector3 { x, y, z };
	}

	pub fn magnitude(&self) -> f32
	{
		self.dot(self).sqrt()
	}

	pub fn distance(&self, r: &Vector3) -> f32
	{
		let dist_x: f32 = self.x - r.x;
		let dist_y: f32 = self.y - r.y;
		let dist_z: f32 = self.z - r.z;

		((dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z)).sqrt()
	}

	pub fn reflect(&self, r: &Vector3) -> Self
	{
		self.sub(&r.mulf(2.0 * self.dot(&r)))
	}

	pub fn rotate_from_vector(&self, axis: &Vector3, angle: f32) -> Vector3
	{
		let sin_angle: f32 = (-angle).sin();
		let cos_angle: f32 = (-angle).cos();

		self.cross(&axis.mulf(sin_angle)).add(
			&(self.mulf(cos_angle)).add(
				&axis.mulf(
					self.dot(
						&axis.mulf(1.0 - cos_angle))
				)
			)
		)
	}

	pub fn rotate_from_quaternion(&self, rotation: &Quaternion) -> Self
	{
		let conjugate = rotation.conjugate();

		let w = rotation.mulv(self).mul(&conjugate);

		Vector3::new(w.x, w.y, w.z)
	}

	pub fn to_euler_rotation_matrix3(&self) -> Matrix3
	{
		let cx = self.x.cos();
		let cy = self.y.cos();
		let cz = self.z.cos();
		let sx = self.x.sin();
		let sy = self.y.sin();
		let sz = self.z.sin();

		let rx =
		[
			[ 1.0, 0.0, 0.0 ],
			[ 0.0, cx, -sx ],
			[ 0.0, sx, cx ]
		];

		let ry =
		[
			[ cy, 0.0, sy ],
			[ 0.0, 1.0, 0.0 ],
			[ -sy, 0.0, cy ]
		];

		let rz =
		[
			[ cz, -sz, 0.0 ],
			[ sz, cz, 0.0 ],
			[ 0.0, 0.0, 1.0 ]
		];

		let mut res = [[0.0; 3]; 3];
		for i in 0..3
		{
			for j in 0..3
			{
				res[i][j] = rx[i][0] * ry[0][j] + rx[i][1] * ry[1][j] + rx[i][2] * ry[2][j];
			}
		}

		for i in 0..3
		{
			for j in 0..3
			{
				res[i][j] = res[i][0] * rz[0][j] + res[i][1] * rz[1][j] + res[i][2] * rz[2][j];
			}
		}

		Matrix3 { val: res }
	}

	fn rotate_matrix3(&self, matrix: &Matrix3) -> Self
	{
		let x: f32 = self.x * matrix.val[0][0] + self.y * matrix.val[0][1] + self.z * matrix.val[0][2];
		let y: f32 = self.x * matrix.val[1][0] + self.y * matrix.val[1][1] + self.z * matrix.val[1][2];
		let z: f32 = self.x * matrix.val[2][0] + self.y * matrix.val[2][1] + self.z * matrix.val[2][2];
		Vector3 { x, y, z }
	}

	pub fn lerp(&self, dest: &Vector3, factor: f32) -> Self
	{
		dest.sub(self).mulf(factor).add(self)
	}

	pub fn abs(&self) -> Self
	{
		Vector3::new(self.x.abs(), self.y.abs(), self.z.abs())
	}

	pub fn atan(&self, r: &Vector3) -> Self
	{
		Self
		{
			x: self.x.atan2(r.x),
			y: self.y.atan2(r.y),
			z: self.z.atan2(r.z)
		}
	}

	pub fn to_radians(&self) -> Self
	{
		Self
		{
			x: self.x.to_radians(),
			y: self.y.to_radians(),
			z: self.z.to_radians()
		}
	}

	pub fn add(&self, r: &Vector3) -> Self
	{
		return Vector3
		{
			x: self.x + r.x,
			y: self.y + r.y,
			z: self.z + r.z
		}
	}

	pub fn sub(&self, r: &Vector3) -> Self
	{
		return Vector3
		{
			x: self.x - r.x,
			y: self.y - r.y,
			z: self.z - r.z
		}
	}

	pub fn mul(&self, r: &Vector3) -> Self
	{
		return Vector3
		{
			x: self.x * r.x,
			y: self.y * r.y,
			z: self.z * r.z
		}
	}

	pub fn div(&self, r: &Vector3) -> Self
	{
		return Vector3
		{
			x: self.x / r.x,
			y: self.y / r.y,
			z: self.z / r.z
		}
	}

	pub fn addf(&self, f: f32) -> Self
	{
		return Vector3
		{
			x: self.x + f,
			y: self.y + f,
			z: self.z + f
		}
	}

	pub fn subf(&self, f: f32) -> Self
	{
		return Vector3
		{
			x: self.x - f,
			y: self.y - f,
			z: self.z - f
		}
	}

	pub fn mulf(&self, f: f32) -> Self
	{
		return Vector3
		{
			x: self.x * f,
			y: self.y * f,
			z: self.z * f
		}
	}

	pub fn divf(&self, f: f32) -> Self
	{
		return Vector3
		{
			x: self.x / f,
			y: self.y / f,
			z: self.z / f
		}
	}

	pub fn xy(&self) -> Vector2
	{
		return Vector2
		{
			x: self.x,
			y: self.y
		}
	}

	pub fn get_forward(&self) -> Self
	{
		let yaw: f32 = self.x.to_radians();
		let pitch: f32 = self.y.to_radians();
		let roll: f32 = self.z.to_radians();

		Self
		{
			x: roll.sin() * pitch.sin() + roll.cos() * yaw.sin() * pitch.cos(),
			y: -roll.cos() * pitch.sin() + roll.sin() * yaw.sin() * pitch.cos(),
			z: yaw.cos() * pitch.cos()
		}.normalized()
	}

	pub fn to_normalized_direction(&self) -> Self
	{
		let (yaw, pitch, roll) = (self.x, self.y, self.z);
		let x: f32 = pitch.sin() * yaw.cos();
    	let y: f32 = pitch.sin() * yaw.sin();
    	let z: f32 = pitch.cos();
		Self { x, y, z }.normalized()
	}

	pub fn print(&self)
	{
		println!("{} {} {}", self.x, self.y, self.z);
	}
}

impl std::ops::Index<usize> for Vector3
{
	type Output = f32;

	fn index(&self, index: usize) -> &Self::Output
	{
		match index
		{
			0 => &self.x,
			1 => &self.y,
			2 => &self.z,
			_ => { panic!("Invalid index!"); }
		}
	}
}

impl std::ops::Add<Vector3> for Vector3
{
	type Output = Vector3;

	fn add(self, r: Vector3) -> Vector3
	{
		Vector3
		{
			x: self.x + r.x,
			y: self.y + r.y,
			z: self.z + r.z
		}
	}
}

impl std::ops::Add<f32> for Vector3
{
	type Output = Vector3;

	fn add(self, r: f32) -> Vector3
	{
		Vector3
		{
			x: self.x + r,
			y: self.y + r,
			z: self.z + r
		}
	}
}

impl std::ops::Sub<Vector3> for Vector3
{
	type Output = Vector3;

	fn sub(self, r: Vector3) -> Vector3
	{
		Vector3
		{
			x: self.x - r.x,
			y: self.y - r.y,
			z: self.z - r.z
		}
	}
}

impl std::ops::Sub<f32> for Vector3
{
	type Output = Vector3;

	fn sub(self, r: f32) -> Vector3
	{
		Vector3
		{
			x: self.x - r,
			y: self.y - r,
			z: self.z - r
		}
	}
}

impl std::ops::Sub<Vector3> for f32
{
	type Output = Vector3;

	fn sub(self, r: Vector3) -> Vector3
	{
		Vector3
		{
			x: self - r.x,
			y: self - r.y,
			z: self - r.z
		}
	}
}

impl std::ops::Mul<Vector3> for Vector3
{
	type Output = Vector3;

	fn mul(self, r: Vector3) -> Vector3
	{
		Vector3
		{
			x: self.x * r.x,
			y: self.y * r.y,
			z: self.z * r.z
		}
	}
}

impl std::ops::Mul<f32> for Vector3
{
	type Output = Vector3;

	fn mul(self, r: f32) -> Vector3
	{
		Vector3
		{
			x: self.x * r,
			y: self.y * r,
			z: self.z * r
		}
	}
}

impl std::ops::Div<Vector3> for Vector3
{
	type Output = Vector3;

	fn div(self, r: Vector3) -> Vector3
	{
		Vector3
		{
			x: self.x / r.x,
			y: self.y / r.y,
			z: self.z / r.z
		}
	}
}

impl std::ops::Div<f32> for Vector3
{
	type Output = Vector3;

	fn div(self, r: f32) -> Vector3
	{
		Vector3
		{
			x: self.x / r,
			y: self.y / r,
			z: self.z / r
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Vector4
{
	pub x: f32,
	pub y: f32,
	pub z: f32,
	pub w: f32
}

impl Vector4
{
	pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self
	{
		Self { x, y, z, w }
	}

	pub fn zero() -> Self
	{
		Self
		{
			x: 0.0,
			y: 0.0,
			z: 0.0,
			w: 0.0
		}
	}

	pub fn addf(&self, f: f32) -> Self
	{
		Self
		{
			x: self.x + f,
			y: self.y + f,
			z: self.z + f,
			w: self.w + f
		}
	}

	pub fn subf(&self, f: f32) -> Self
	{
		Self
		{
			x: self.x - f,
			y: self.y - f,
			z: self.z - f,
			w: self.w - f
		}
	}

	pub fn mulf(&self, f: f32) -> Self
	{
		Self
		{
			x: self.x * f,
			y: self.y * f,
			z: self.z * f,
			w: self.w * f
		}
	}

	pub fn divf(&self, f: f32) -> Self
	{
		Self
		{
			x: self.x / f,
			y: self.y / f,
			z: self.z / f,
			w: self.w / f
		}
	}

	pub fn xyz(&self) -> Vector3
	{
		Vector3 { x: self.x, y: self.y, z: self.z }
	}
}

impl std::ops::Index<usize> for Vector4
{
	type Output = f32;

	fn index(&self, index: usize) -> &Self::Output
	{
		match index
		{
			0 => &self.x,
			1 => &self.y,
			2 => &self.z,
			3 => &self.w,
			_ => { panic!("Invalid index!"); }
		}
	}
}

impl std::ops::Add<Vector4> for Vector4
{
	type Output = Vector4;

	fn add(self, r: Vector4) -> Vector4
	{
		Vector4
		{
			x: self.x + r.x,
			y: self.y + r.y,
			z: self.z + r.z,
			w: self.w + r.w
		}
	}
}

impl std::ops::Sub<Vector4> for Vector4
{
	type Output = Vector4;

	fn sub(self, r: Vector4) -> Vector4
	{
		Vector4
		{
			x: self.x - r.x,
			y: self.y - r.y,
			z: self.z - r.z,
			w: self.w - r.w
		}
	}
}

impl std::ops::Mul<Vector4> for Vector4
{
	type Output = Vector4;

	fn mul(self, r: Vector4) -> Vector4
	{
		Vector4
		{
			x: self.x * r.x,
			y: self.y * r.y,
			z: self.z * r.z,
			w: self.w * r.w
		}
	}
}

impl std::ops::Div<Vector4> for Vector4
{
	type Output = Vector4;

	fn div(self, r: Vector4) -> Vector4
	{
		Vector4
		{
			x: self.x / r.x,
			y: self.y / r.y,
			z: self.z / r.z,
			w: self.w / r.w
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Quaternion
{
	pub x: f32,
	pub y: f32,
	pub z: f32,
	pub w: f32
}

impl Quaternion
{
	pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self
	{
		Self { x, y, z, w }
	}

	pub fn from_axis(axis: &Vector3, angle: f32) -> Self
	{
		let sin_half_angle = (angle / 2.0).sin();
		let cos_half_angle = (angle / 2.0).cos();

		Self
		{
			x: axis.x * sin_half_angle,
			y: axis.y * sin_half_angle,
			z: axis.z * sin_half_angle,
			w: cos_half_angle
		}
	}

	pub fn from_rotation_x(angle: f32) -> Self
	{
		Self::from_axis(&Vector3::new(1.0, 0.0, 0.0), angle)
	}

	pub fn from_rotation_y(angle: f32) -> Self
	{
		Self::from_axis(&Vector3::new(0.0, 1.0, 0.0), angle)
	}

	pub fn from_rotation_z(angle: f32) -> Self
	{
		Self::from_axis(&Vector3::new(0.0, 0.0, 1.0), angle)
	}

	pub fn yaw(&self) -> f32
	{
		let test: f32 = self.x * self.y + self.z * self.w;
		if test > 0.499
		{
			2.0 * (self.x * self.z - self.y * self.w).atan()
		}
		else if test < -0.499
		{
			-2.0 * (self.x * self.z - self.y * self.w).atan()
		}
		else
		{
			(self.x * self.w + self.y * self.z).atan2(1.0 - 2.0 * (self.x * self.x + self.y * self.y))
		}
	}

	pub fn to_euler_vector3(&self) -> Vector3
	{
		let sqrx: f32 = self.x * self.x;
		let sqry: f32 = self.y * self.y;
		let sqrz: f32 = self.z * self.z;
		let sqrw: f32 = self.w * self.w;

		Vector3
		{
			x: (-2.0 * (self.x * self.z - self.y * self.w)).asin(),
			y: ( 2.0 * (self.y * self.z + self.x * self.w)).atan2(-sqrx - sqry + sqrz + sqrw),
			z: ( 2.0 * (self.x * self.y + self.z * self.w)).atan2( sqrx - sqry - sqrz + sqrw)
		}
	}

	pub fn from_euler_vector3(r: &Vector3) -> Self
	{
		let fc1: f32 = (r.z / 2.0).cos();
		let fc2: f32 = (r.x / 2.0).cos();
		let fc3: f32 = (r.y / 2.0).cos();

		let fs1: f32 = (r.z / 2.0).sin();
		let fs2: f32 = (r.x / 2.0).sin();
		let fs3: f32 = (r.y / 2.0).sin();

		Self
		{
			x: fc1 * fc2 * fs3 - fs1 * fs2 * fc3,
			y: fc1 * fs2 * fc3 + fs1 * fc2 * fs3,
			z: fs1 * fc2 * fc3 - fc1 * fs2 * fs3,
			w: fc1 * fc2 * fc3 + fs1 * fs2 * fs3
		}
	}

	pub fn from_rotation_matrix(rot: &Matrix4) -> Self
	{
		let trace = rot.get(0, 0) + rot.get(1, 1) + rot.get(2, 2);

		let mut x: f32 = 0.0;
		let mut y: f32 = 0.0;
		let mut z: f32 = 0.0;
		let mut w: f32 = 0.0;

		if trace > 0.0
		{
			let s = 0.5 / (trace + 1.0).sqrt();
			w = 0.25 / s;
			x = (rot.get(1, 2) - rot.get(2, 1)) * s;
			y = (rot.get(2, 0) - rot.get(0, 2)) * s;
			z = (rot.get(0, 1) - rot.get(1, 0)) * s;
		}
		else
		{
			if rot.get(0, 0) > rot.get(1, 1) && rot.get(0, 0) > rot.get(2, 2)
			{
				let s = 2.0 * (1.0 + rot.get(0, 0) - rot.get(1, 1) - rot.get(2, 2)).sqrt();
				w = (rot.get(1, 2) - rot.get(2, 1)) / s;
				x = 0.25 * s;
				y = (rot.get(1, 0) + rot.get(0, 1)) / s;
				z = (rot.get(2, 0) + rot.get(0, 2)) / s;
			}
			else if rot.get(1, 1) > rot.get(2, 2)
			{
				let s = 2.0 * (1.0 + rot.get(1, 1) - rot.get(0, 0) - rot.get(2, 2)).sqrt();
				w = (rot.get(2, 0) - rot.get(0, 2)) / s;
				x = (rot.get(1, 0) + rot.get(0, 1)) / s;
				y = 0.25 * s;
				z = (rot.get(2, 1) + rot.get(1, 2)) / s;
			}
			else
			{
				let s = 2.0 * (1.0 + rot.get(2, 2) - rot.get(0, 0) - rot.get(1, 1)).sqrt();
				w = (rot.get(0, 1) - rot.get(1, 0)) / s;
				x = (rot.get(2, 0) + rot.get(0, 2)) / s;
				y = (rot.get(1, 2) + rot.get(2, 1)) / s;
				z = 0.25 * s;
			}
		}

		let length = (x * x + y * y + z * z + w * w).sqrt();
		x /= length;
		y /= length;
		z /= length;
		w /= length;

		Quaternion { x, y, z, w }
	}

	pub fn length(&self) -> f32
	{
		return (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
	}

	pub fn normalized(&self) -> Quaternion
	{
		let length: f32 = self.length();

		return Quaternion {
			x: self.x / length,
			y: self.y / length,
			z: self.z / length,
			w: self.w / length
		};
	}

	pub fn conjugate(&self) -> Self
	{
		Self
		{
			x: -self.x,
			y: -self.y,
			z: -self.z,
			w: self.w
		}
	}

	pub fn mulf(&self, r: f32) -> Self
	{
		Self
		{
			x: self.x * r,
			y: self.y * r,
			z: self.z * r,
			w: self.w * r
		}
	}

	pub fn mul(&self, r: &Quaternion) -> Self
	{
		Self
		{
			x: self.x * r.w + self.w * r.x + self.y * r.z - self.z * r.y,
			y: self.y * r.w + self.w * r.y + self.z * r.x - self.x * r.z,
			z: self.z * r.w + self.w * r.z + self.x * r.y - self.y * r.x,
			w: self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z
		}
	}

	pub fn mulv(&self, r: &Vector3) -> Self
	{
		Self
		{
			x:  self.w * r.x + self.y * r.z - self.z * r.y,
			y:  self.w * r.y + self.z * r.x - self.x * r.z,
			z:  self.w * r.z + self.x * r.y - self.y * r.x,
			w: -self.x * r.x - self.y * r.y - self.z * r.z
		}
	}

	pub fn add(&self, r: &Quaternion) -> Self
	{
		Self
		{
			x: self.x + r.x,
			y: self.y + r.y,
			z: self.z + r.z,
			w: self.w + r.w
		}
	}

	pub fn sub(&self, r: &Quaternion) -> Self
	{
		Self
		{
			x: self.x - r.x,
			y: self.y - r.y,
			z: self.z - r.z,
			w: self.w - r.w
		}
	}

	pub fn to_rotation_matrix(&self) -> Matrix4
	{
		let forward =  Vector3::new(2.0 * (self.x * self.z - self.w * self.y), 2.0 * (self.y * self.z + self.w * self.x), 1.0 - 2.0 * (self.x * self.x + self.y * self.y));
		let up = Vector3::new(2.0 * (self.x * self.y + self.w * self.z), 1.0 - 2.0 * (self.x * self.x + self.z * self.z), 2.0 * (self.y * self.z - self.w * self.x));
		let right = Vector3::new(1.0 - 2.0 * (self.y * self.y + self.z * self.z), 2.0 * (self.x * self.y - self.w * self.z), 2.0 * (self.x * self.z + self.w * self.y));

		Matrix4::init_rotation_from_vec(&forward, &up, &right)
	}

	pub fn dot(&self, r: &Quaternion) -> f32
	{
		self.x * r.x + self.y * r.y + self.z * r.z + self.w * r.w
	}

	pub fn nlerp(&self, dest: &Quaternion, lerp_factor: f32, shortest: bool) -> Quaternion
	{
		let mut corrected_dest = dest.clone();

		if shortest && self.dot(&dest) < 0.0
		{
			corrected_dest = Quaternion::new(-dest.x, -dest.y, -dest.z, -dest.w);
		}

		corrected_dest.sub(self).mulf(lerp_factor).add(self).normalized()
	}

	pub fn slerp(&self, dest: &Quaternion, lerp_factor: f32, shortest: bool) -> Self
	{
		const EPSILON: f32 = 1e3;

		let mut c = self.dot(dest);
		let mut corrected_dest = dest.clone();

		if shortest && c < 0.0
		{
			c = -c;
			corrected_dest = Quaternion::new(-dest.x, -dest.y, -dest.z, -dest.w);
		}

		if c.abs() >= 1.0 - EPSILON
		{
			return self.nlerp(&corrected_dest, lerp_factor, false);
		}

		let s = (1.0 - c * c).sqrt();
		let angle = s.atan2(c);
		let inv_s = 1.0 / s;

		let src_factor = ((1.0 - lerp_factor) * angle).sin() * inv_s;
		let dest_factor = (lerp_factor * angle).sin() * inv_s;

		self.mulf(src_factor).add(&corrected_dest.mulf(dest_factor))
	}

	pub fn get_forward(&self) -> Vector3
	{
		Vector3::new(0.0, 0.0, 1.0).rotate_from_quaternion(self)
	}

	pub fn get_back(&self) -> Vector3
	{
		Vector3::new(0.0, 0.0, -1.0).rotate_from_quaternion(self)
	}

	pub fn get_up(&self) -> Vector3
	{
		Vector3::new(0.0, 1.0, 0.0).rotate_from_quaternion(self)
	}

	pub fn get_down(&self) -> Vector3
	{
		Vector3::new(0.0, -1.0, 0.0).rotate_from_quaternion(self)
	}

	pub fn get_right(&self) -> Vector3
	{
		Vector3::new(1.0, 0.0, 0.0).rotate_from_quaternion(self)
	}

	pub fn get_left(&self) -> Vector3
	{
		Vector3::new(-1.0, 0.0, 0.0).rotate_from_quaternion(self)
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Matrix3
{
	pub val: [[f32; 3]; 3]
}

impl Matrix3
{
	pub fn new(xx: f32, xy: f32, xz: f32,
		yx: f32, yy: f32, yz: f32,
		zx: f32, zy: f32, zz: f32) -> Self
	{
		let val: [[f32; 3]; 3] =
		[
			[ xx, xy, xz ],
			[ yx, yy, yz ],
			[ zx, zy, zz ]
		];

		Self { val }
	}

	pub fn from_array(values: &[f32; 9]) -> Self
	{
		let val: [[f32; 3]; 3] =
		[
			[ values[0], values[1], values[2] ],
			[ values[3], values[4], values[5] ],
			[ values[6], values[7], values[8] ]
		];

		Self { val }
	}

	pub fn identity(&self) -> Self
	{
		let val: [[f32; 3]; 3] =
		[
			[ 1.0, 0.0, 0.0 ],
			[ 0.0, 1.0, 0.0 ],
			[ 0.0, 0.0, 1.0 ],
		];

		Self { val }
	}

	pub fn transpose(&self) -> Self
	{
		let val: [[f32; 3]; 3] =
		[
			[ self.val[0][0], self.val[1][0], self.val[2][0] ],
			[ self.val[0][1], self.val[1][1], self.val[2][1] ],
			[ self.val[0][2], self.val[1][2], self.val[2][2] ],
		];

		Self { val }
	}

	pub fn det(&self) -> f32
	{
		(self.val[0][0] * self.val[1][1] * self.val[2][2]) + (self.val[0][1] * self.val[1][2] * self.val[2][0]) + (self.val[0][2] * self.val[1][0] * self.val[2][1]) -
        (self.val[0][2] * self.val[1][1] * self.val[2][0]) - (self.val[0][1] * self.val[1][0] * self.val[2][2]) - (self.val[0][0] * self.val[1][2] * self.val[2][1])
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Matrix4
{
	pub val: [[f32; 4]; 4]
}

impl Matrix4
{
	pub fn zero() -> Self
	{
		let val: [[f32; 4]; 4] =
		[
			[ 0.0; 4 ],
			[ 0.0; 4 ],
			[ 0.0; 4 ],
			[ 0.0; 4 ]
		];

		Self { val }
	}

	pub fn identity() -> Self
	{
		let val: [[f32; 4]; 4] =
		[
			[ 1.0, 0.0, 0.0, 0.0 ],
			[ 0.0, 1.0, 0.0, 0.0 ],
			[ 0.0, 0.0, 1.0, 0.0 ],
			[ 0.0, 0.0, 0.0, 1.0 ]
		];

		Self { val }
	}

	pub fn init_translation(x: f32, y: f32, z: f32) -> Self
	{
		let val: [[f32; 4]; 4] =
		[
			[ 1.0, 0.0, 0.0, x ],
			[ 0.0, 1.0, 0.0, y ],
			[ 0.0, 0.0, 1.0, z ],
			[ 0.0, 0.0, 0.0, 1.0 ]
		];

		Self { val }
	}

	pub fn init_rotation(mut x: f32, mut y: f32, mut z: f32) -> Self
	{
		let mut rx = Matrix4::identity();
		let mut ry = Matrix4::identity();
		let mut rz = Matrix4::identity();

		x = x.to_radians();
		y = y.to_radians();
		z = z.to_radians();

		rz.val[0][0] = z.cos(); rz.val[0][1] = -z.cos(); rz.val[0][2] = 0.0; rz.val[0][3] = 0.0;
		rz.val[1][0] = z.sin(); rz.val[1][1] =  z.cos(); rz.val[1][2] = 0.0; rz.val[1][3] = 0.0;
		rz.val[2][0] = 0.0;     rz.val[2][1] = 0.0;      rz.val[2][2] = 1.0; rz.val[2][3] = 0.0;
		rz.val[3][0] = 0.0;     rz.val[3][1] = 0.0;      rz.val[3][2] = 0.0; rz.val[3][3] = 1.0;

		rx.val[0][0] = 1.0; rx.val[0][1] = 0.0;     rx.val[0][2] = 0.0;      rx.val[0][3] = 0.0;
		rx.val[1][0] = 0.0; rx.val[1][1] = x.cos(); rx.val[1][2] = -x.sin(); rx.val[1][3] = 0.0;
		rx.val[2][0] = 0.0; rx.val[2][1] = x.sin(); rx.val[2][2] = x.cos();  rx.val[2][3] = 0.0;
		rx.val[3][0] = 0.0; rx.val[3][1] = 0.0;     rx.val[3][2] = 0.0;      rx.val[3][3] = 1.0;

		ry.val[0][0] = y.cos(); ry.val[0][1] = 0.0; ry.val[0][2] = -y.sin(); ry.val[0][3] = 0.0;
		ry.val[1][0] = 0.0;     ry.val[1][1] = 1.0; ry.val[1][2] = 0.0;      ry.val[1][3] = 0.0;
		ry.val[2][0] = y.sin(); ry.val[2][1] = 0.0; ry.val[2][2] = y.cos();  ry.val[2][3] = 0.0;
		ry.val[3][0] = 0.0;     ry.val[3][1] = 0.0; ry.val[3][2] = 0.0;      ry.val[3][3] = 1.0;

		rz.mul(&ry.mul(&rx))
	}

	pub fn init_scale(x: f32, y: f32, z: f32) -> Self
	{
		let val: [[f32; 4]; 4] =
		[
			[   x, 0.0, 0.0, 0.0 ],
			[ 0.0,   y, 0.0, 0.0 ],
			[ 0.0, 0.0,   z, 0.0 ],
			[ 0.0, 0.0, 0.0, 1.0 ]
		];

		Self { val }
	}

	// note: this is in row-major order
	pub fn init_perspective(fov: f32, aspect: f32, z_near: f32, z_far: f32) -> Self
	{
		let tan_half_fov = ((fov / 2.0).to_radians()).tan();
		let z_range = z_near - z_far;

		let val: [[f32; 4]; 4] =
		[
			[ 1.0 / (tan_half_fov * aspect), 0.0,                0.0,                         0.0 ],
			[ 0.0,                           1.0 / tan_half_fov, 0.0,                         0.0 ],
			[ 0.0,                           0.0,                (-z_near - z_far) / z_range, 2.0 * z_far * z_near / z_range ],
			[ 0.0,                           0.0,                1.0,                         0.0 ]
		];

		Self { val }
	}

	// note: this is in row-major order
	pub fn init_orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self
	{
		let width = right - left;
		let height = top - bottom;
		let depth = far - near;

		let val: [[f32; 4]; 4] =
		[
			[ 2.0 / width, 0.0,           0.0,         -(right + left) / width ],
			[ 0.0,         2.0 / height,  0.0,         -(top + bottom) / height ],
			[ 0.0,         0.0,          -2.0 / depth, -(far + near) / depth ],
			[ 0.0,         0.0,           0.0,          1.0 ]
		];

		Self { val }
	}

	#[deprecated]
	pub fn init_view(eye: &Vector3, center: &Vector3, up: &Vector3) -> Self
	{
		let mut Z = eye.sub(center).normalized();
		let mut Y = up.clone();
		let mut X = Y.cross(&Z);

		Y = Z.cross(&X);
		X = X.normalized();
		Y = Y.normalized();

		let val: [[f32; 4]; 4] =
		[
			[ X.x, Y.x, Z.x, 0.0 ],
			[ X.y, Y.y, Z.y, 0.0 ],
			[ X.z, Y.z, Z.z, 0.0 ],
			[ -X.dot(eye), -Y.dot(eye), -Z.dot(eye), 1.0 ]
		];

		Self { val }
	}

	pub fn init_rotation_from_direction(forward: &Vector3, up: &Vector3) -> Self
	{
		let f = forward.normalized();

		let mut r = up.normalized();
		r = r.cross(&f);

		let u = f.cross(&r);

		Matrix4::init_rotation_from_vec(&f, &u, &r)
	}

	pub fn init_rotation_from_vec(forward: &Vector3, up: &Vector3, right: &Vector3) -> Self
	{
		let val: [[f32; 4]; 4] =
		[
			[ right.x, right.y, right.z, 0.0 ],
			[ up.x, up.y, up.z, 0.0 ],
			[ forward.x, forward.y, forward.z, 0.0 ],
			[ 0.0, 0.0, 0.0, 1.0 ]
		];

		Self { val }
	}

	pub fn init_billboard(pos: &Vector3, camera_pos: &Vector3) -> Self
	{
		let look = camera_pos.sub(pos).normalized();
		let up = Vector3::new(0.0, 1.0, 0.0).normalized();
		let right = up.cross(&look).normalized();

		let val: [[f32; 4]; 4] =
		[
			[ right.x, right.y, right.z, 0.0 ],
			[ up.x,    up.y,    up.z,    0.0 ],
			[ look.x,  look.y,  look.z,  0.0 ],
			[ pos.x,   pos.y,   pos.z,   1.0 ]
		];

		Self { val }
	}

	pub fn get(&self, i: usize, j: usize) -> f32
	{
		self.val[i][j]
	}

	pub fn set(&mut self, i: usize, j: usize, val: f32)
	{
		self.val[i][j] = val;
	}

	pub fn column(&self, index: usize) -> Vector4
	{
        Vector4::new(self.val[0][index], self.val[1][index], self.val[2][index], self.val[3][index])
    }

	pub fn transpose(&self) -> Self
	{
		let val: [[f32; 4]; 4] =
		[
			[ self.val[0][0], self.val[1][0], self.val[2][0], self.val[3][0] ],
			[ self.val[0][1], self.val[1][1], self.val[2][1], self.val[3][1] ],
			[ self.val[0][2], self.val[1][2], self.val[2][2], self.val[3][2] ],
			[ self.val[0][3], self.val[1][3], self.val[2][3], self.val[3][3] ]
		];

		Self { val }
	}

	pub fn transform(&self, r: &Vector3) -> Vector3
	{
		Vector3
		{
			x: self.val[0][0] * r.x + self.val[0][1] * r.y + self.val[0][2] * r.z + self.val[0][3],
			y: self.val[1][0] * r.x + self.val[1][1] * r.y + self.val[1][2] * r.z + self.val[1][3],
			z: self.val[2][0] * r.x + self.val[2][1] * r.y + self.val[2][2] * r.z + self.val[2][3]
		}
	}

	pub fn mul(&self, r: &Matrix4) -> Self
	{
		let mut result = Matrix4::zero();

		for i in 0..4
		{
			for j in 0..4
			{
				result.set(i, j, self.get(i, 0) * r.get(0, j) +
									  self.get(i, 1) * r.get(1, j) +
									  self.get(i, 2) * r.get(2, j) +
									  self.get(i, 3) * r.get(3, j));
			}
		}

		result
	}

	pub fn mul_vec3(&self, v: &Vector3) -> Vector3
	{
		todo!()
	}

	pub fn det(&self) -> f32
	{
		let cofact_xx: f32 =  (Matrix3::new(self.val[1][1], self.val[1][2], self.val[1][3], self.val[2][1], self.val[2][2], self.val[2][3], self.val[3][1], self.val[3][2], self.val[3][3])).det();
		let cofact_xy: f32 = -(Matrix3::new(self.val[1][0], self.val[1][2], self.val[1][3], self.val[2][0], self.val[2][2], self.val[2][3], self.val[3][0], self.val[3][2], self.val[3][3])).det();
		let cofact_xz: f32 =  (Matrix3::new(self.val[1][0], self.val[1][1], self.val[1][3], self.val[2][0], self.val[2][1], self.val[2][3], self.val[3][0], self.val[3][1], self.val[3][3])).det();
		let cofact_xw: f32 = -(Matrix3::new(self.val[1][0], self.val[1][1], self.val[1][2], self.val[2][0], self.val[2][1], self.val[2][2], self.val[3][0], self.val[3][1], self.val[3][2])).det();
		
		return (cofact_xx * self.val[0][0]) + (cofact_xy * self.val[0][1]) + (cofact_xz * self.val[0][2]) + (cofact_xw * self.val[0][3]);
	}

	pub fn inverse(&self) -> Self
	{
		let det: f32 = self.det();
		let fac: f32 = 1.0 / det;

		let val: [[f32; 4]; 4] =
		[
			[
				fac *  (Matrix3::new(self.val[1][1], self.val[1][2], self.val[1][3], self.val[2][1], self.val[2][2], self.val[2][3], self.val[3][1], self.val[3][2], self.val[3][3])).det(),
				fac * -(Matrix3::new(self.val[1][0], self.val[1][2], self.val[1][3], self.val[2][0], self.val[2][2], self.val[2][3], self.val[3][0], self.val[3][2], self.val[3][3])).det(),
				fac *  (Matrix3::new(self.val[1][0], self.val[1][1], self.val[1][3], self.val[2][0], self.val[2][1], self.val[2][3], self.val[3][0], self.val[3][1], self.val[3][3])).det(),
				fac * -(Matrix3::new(self.val[1][0], self.val[1][1], self.val[1][2], self.val[2][0], self.val[2][1], self.val[2][2], self.val[3][0], self.val[3][1], self.val[3][2])).det()
			],

			[
				fac * -(Matrix3::new(self.val[0][1], self.val[0][2], self.val[0][3], self.val[2][1], self.val[2][2], self.val[2][3], self.val[3][1], self.val[3][2], self.val[3][3])).det(),
				fac *  (Matrix3::new(self.val[0][0], self.val[0][2], self.val[0][3], self.val[2][0], self.val[2][2], self.val[2][3], self.val[3][0], self.val[3][2], self.val[3][3])).det(),
				fac * -(Matrix3::new(self.val[0][0], self.val[0][1], self.val[0][3], self.val[2][0], self.val[2][1], self.val[2][3], self.val[3][0], self.val[3][1], self.val[3][3])).det(),
				fac *  (Matrix3::new(self.val[0][0], self.val[0][1], self.val[0][2], self.val[2][0], self.val[2][1], self.val[2][2], self.val[3][0], self.val[3][1], self.val[3][2])).det()
			],

			[
				fac *  (Matrix3::new(self.val[0][1], self.val[0][2], self.val[0][3], self.val[1][1], self.val[1][2], self.val[1][3], self.val[3][1], self.val[3][2], self.val[3][3])).det(),
				fac * -(Matrix3::new(self.val[0][0], self.val[0][2], self.val[0][3], self.val[1][0], self.val[1][2], self.val[1][3], self.val[3][0], self.val[3][2], self.val[3][3])).det(),
				fac *  (Matrix3::new(self.val[0][0], self.val[0][1], self.val[0][3], self.val[1][0], self.val[1][1], self.val[1][3], self.val[3][0], self.val[3][1], self.val[3][3])).det(),
				fac * -(Matrix3::new(self.val[0][0], self.val[0][1], self.val[0][2], self.val[1][0], self.val[1][1], self.val[1][2], self.val[3][0], self.val[3][1], self.val[3][2])).det()
			],

			[
				fac * -(Matrix3::new(self.val[0][1], self.val[0][2], self.val[0][3], self.val[1][1], self.val[1][2], self.val[1][3], self.val[2][1], self.val[2][2], self.val[2][3])).det(),
				fac *  (Matrix3::new(self.val[0][0], self.val[0][2], self.val[0][3], self.val[1][0], self.val[1][2], self.val[1][3], self.val[2][0], self.val[2][2], self.val[2][3])).det(),
				fac * -(Matrix3::new(self.val[0][0], self.val[0][1], self.val[0][3], self.val[1][0], self.val[1][1], self.val[1][3], self.val[2][0], self.val[2][1], self.val[2][3])).det(),
				fac *  (Matrix3::new(self.val[0][0], self.val[0][1], self.val[0][2], self.val[1][0], self.val[1][1], self.val[1][2], self.val[2][0], self.val[2][1], self.val[2][2])).det()
			]
		];

		Self { val }
	}

	pub fn get_ptr(&self) -> *const f32
	{
		self.val.as_ptr() as *const f32
	}

	pub fn to_array(&self) -> [f32; 16]
	{
		[
			self.val[0][0], self.val[0][1], self.val[0][2], self.val[0][3],
			self.val[1][0], self.val[1][1], self.val[1][2], self.val[1][3],
			self.val[2][0], self.val[2][1], self.val[2][2], self.val[2][3],
			self.val[3][0], self.val[3][1], self.val[3][2], self.val[3][3]
		]
	}
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Transform
{
	pub translation: Vector3,
	pub rotation: Quaternion,
	pub scale: Vector3
}

impl Transform
{
	pub fn new() -> Self
	{
		Self
		{
			translation: Vector3::new(0.0, 0.0, 0.0),
			rotation: Quaternion::new(0.0, 0.0, 0.0, 1.0),
			scale: Vector3::new(1.0, 1.0, 1.0),
		}
	}

	pub fn from_data(translation: &Vector3, rotation: &Quaternion, scale: &Vector3) -> Self
	{
		Self
		{
			translation: *translation,
			rotation: *rotation,
			scale: *scale
		}
	}

	pub fn rotate(&mut self, axis: &Vector3, angle: f32)
	{
		self.rotation = Quaternion::from_axis(axis, angle).mul(&self.rotation).normalized();
	}

	pub fn look_at(&mut self, point: &Vector3, up: &Vector3)
	{
		self.rotation = self.get_look_at_rotation(point, up);
	}

	pub fn get_look_at_rotation(&self, point: &Vector3, up: &Vector3) -> Quaternion
	{
		let forward = point.sub(&self.translation).normalized();
		Quaternion::from_rotation_matrix(&Matrix4::init_rotation_from_direction(&forward, up))
	}

	pub fn get_transformation(&self) -> Matrix4
	{
		let translation_matrix = Matrix4::init_translation(self.translation.x, self.translation.y, self.translation.z);
		let rotation_matrix = self.rotation.to_rotation_matrix();
		let scale_matrix = Matrix4::init_scale(self.scale.x, self.scale.y, self.scale.z);

		translation_matrix.mul(&rotation_matrix.mul(&scale_matrix))
	}
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct AABB
{
	pub min: Vector3,
	pub max: Vector3
}

impl AABB
{
	pub fn new(min: &Vector3, max: &Vector3) -> Self
	{
		Self { min: min.clone(), max: max.clone() }
	}

	pub fn offset(&self, x: f32, y: f32, z: f32) -> AABB
	{
		Self
		{
			min: Vector3
			{
				x: self.min.x + x,
				y: self.min.y + y,
				z: self.min.z + z
			},
			max: Vector3
			{
				x: self.min.x + x,
				y: self.min.y + y,
				z: self.min.z + z
			}
		}
	}

	pub fn intersects_aabb(&self, other: &AABB) -> bool
	{
		todo!()
	}
}

#[derive(Debug, Clone)]
pub struct Random
{
	rng: rand::rngs::ThreadRng
}

impl Random
{
	pub fn new(seed: u64) -> Self
	{
		let mut rng = rand::thread_rng();

		Self { rng }
	}

	pub fn next_f32(&mut self) -> f32
	{
		self.rng.gen::<f32>()
	}

	pub fn next_f64(&mut self) -> f64
	{
		self.rng.gen::<f64>()
	}

	pub fn next_u32(&mut self) -> u32
	{
		self.rng.gen::<u32>()
	}

	pub fn next_u64(&mut self) -> u64
	{
		self.rng.gen::<u64>()
	}

	pub fn next_bool(&mut self) -> bool
	{
		self.rng.gen::<bool>()
	}
}

#[derive(Copy, Clone, Debug)]
pub struct SimplexNoise
{
	pub perm: [i32; 512]
}

impl SimplexNoise
{
	const PERM_SIMPLEX: [i32; 512] =
	[
		151,160,137,91,90,15,
		131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
		190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
		88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
		77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
		102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
		135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
		5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
		223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
		129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
		251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
		49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
		151,160,137,91,90,15,
		131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
		190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
		88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
		77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
		102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
		135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
		5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
		223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
		129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
		251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
		49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
	];

	pub fn new(seed: u64) -> Self
	{
		let mut rnd = Random::new(seed);

		let mut perm: [i32; 512] = Self::PERM_SIMPLEX;
		for i in 0..1024 as usize
		{
			let index0: usize = (rnd.next_u32() % 256) as usize;
			let index1: usize = (rnd.next_u32() % 256) as usize;

			let tmp: i32 = perm[index0];
			perm[index0] = perm[index1];
			perm[index1] = tmp;
		}

		Self { perm }
	}

	fn grad2(hash: i32, x: f64, y: f64) -> f64
	{
		let h: i32 = hash & 7;
		let u: f64 = if h < 4 { x } else { y };
		let v: f64 = if h < 4 { x } else { y };
		(if (h & 1) != 0 { -u } else { u }) + (if (h & 2) != 0 { -2.0 * v } else { 2.0 * v })
	}

	pub fn gen(&self, x: f64, y: f64) -> f64
	{
		let F2: f64 = 0.5 * ((3.0 as f64).sqrt() - 1.0);
		let G2: f64 = (3.0 - (3.0 as f64).sqrt()) / 6.0;
		let F3: f64 = 1.0 / 3.0;
		let G3: f64 = 1.0 / 6.0;
		let F4: f64 = ((5.0 as f64).sqrt() - 1.0) / 4.0;
		let G4: f64 = (5.0 - (5.0 as f64).sqrt()) / 20.0;

		let mut n0: f64 = 0.0;
		let mut n1: f64 = 0.0;
		let mut n2: f64 = 0.0;

		let s: f64 = (x + y) * F2;
		let xs: f64 = x + s;
		let ys: f64 = y + s;
		let i: i32 = xs.floor() as i32;
		let j: i32 = ys.floor() as i32;

		let t: f64 = ((i + j) as f64) * G2;
		let X0: f64 = (i as f64) - t;
		let Y0: f64 = (j as f64) - t;
		let x0: f64 = x - X0;
		let y0: f64 = y - Y0;

		let mut i1: i32 = 0;
		let mut j1: i32 = 0;
		if x0 > y0 { i1 = 1; j1 = 0; }
		else { i1 = 0; j1 = 1; }

		let x1: f64 = x0 - (i1 as f64) + G2;
		let y1: f64 = y0 - (j1 as f64) + G2;
		let x2: f64 = x0 - 1.0 + 2.0 * G2;
		let y2: f64 = y0 - 1.0 + 2.0 * G2;

		let ii: i32 = i % 256;
		let jj: i32 = j % 256;
		dbg!(&ii);
		dbg!(&jj);

		let mut t0: f64 = 0.5 - x0 * x0 - y0 * y0;
		if t0 < 0.0 { n0 = 0.0; }
		else
		{
			t0 *= t0;
			n0 = t0 * t0 * Self::grad2(self.perm[(ii + self.perm[jj as usize]) as usize], x0, y0);
		}

		let mut t1: f64 = 0.5 * x1 - y1 * y1;
		if t < 0.0 { n1 = 0.0 }
		else
		{
			t1 *= t1;
			n1 = t1 * t1 * Self::grad2(self.perm[(ii + i1 + self.perm[(jj + j1) as usize]) as usize], x1, y1);
		}

		let mut t2: f64 = 0.5 - x2 * x2 - y2 * y2;
		if t2 < 0.0 { n2 = 0.0; }
		else
		{
			t2 *= t2;
			n2 = t2 * t2 * Self::grad2(self.perm[(ii + 1 + self.perm[(jj + 1) as usize]) as usize], x2, y2);
		}

		40.0 * (n0 + n1 + n2)
	}
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct AxisAlignedBoundingBox
{
	pub min: Vector3,
	pub max: Vector3
}

impl AxisAlignedBoundingBox
{
	pub fn new(min: Vector3, max: Vector3) -> Self
	{
		Self { min, max }
	}

	pub fn translate(&self, x: f32, y: f32, z: f32) -> Self
	{
		Self
		{
			min: self.min.add(&Vector3::new(x, y, z)),
			max: self.max.add(&Vector3::new(x, y, z))
		}
	}

	pub fn intersects(&self, other: &AxisAlignedBoundingBox) -> bool
	{
		(self.min.x < other.max.x && self.max.x > other.min.x) &&
			(self.min.y < other.max.y && self.max.y > other.min.y) &&
			(self.min.z < other.max.z && self.max.z > other.min.z)
	}

	pub fn contains_point(&self, p: &Vector3) -> bool
	{
		(self.min.x < p.x && self.max.x > p.x) &&
            (self.min.y < p.y && self.max.y > p.y) &&
            (self.min.z < p.z && self.max.z > p.z)
	}

	pub fn clip_line(&self, d: usize, v0: &Vector3, v1: &Vector3, f_low: &mut f32, f_high: &mut f32) -> bool
	{
		let mut f_dim_low: f32 = (self.min[d] - v0[d]) / (v1[d] - v0[d]);
		let mut f_dim_high: f32 = (self.max[d] - v0[d]) / (v1[d] - v0[d]);

		if f_dim_high < f_dim_low
		{
			std::mem::swap(&mut f_dim_high, &mut f_dim_low);
		}

		if f_dim_high < *f_low
		{
			return false;
		}

		if f_dim_low > *f_high
		{
			return false;
		}

		*f_low = f_dim_low.max(*f_low);
		*f_high = f_dim_high.max(*f_high);

		if f_low > f_high
		{
			return false;
		}

		true
	}

	pub fn intersects_line(&self, v0: &Vector3, v1: &Vector3) -> bool
	{
		let mut f_low: f32 = 0.0;
		let mut f_high: f32 = 1.0;

		if !self.clip_line(0, v0, v1, &mut f_low, &mut f_high)
		{
			return false;
		}

		if !self.clip_line(1, v0, v1, &mut f_low, &mut f_high)
		{
			return false;
		}

		if !self.clip_line(2, v0, v1, &mut f_low, &mut f_high)
		{
			return false;
		}

		let b = v1.sub(v0);
		let intersection = *v0 + b * f_low;

		let fl_fraction = f_low;

		true
	}
}

pub fn get_closest_aabb(position: &Vector3, bounding_boxes: &[AxisAlignedBoundingBox]) -> AxisAlignedBoundingBox
{
	let mut bounding_boxes_sorted: Vec<AxisAlignedBoundingBox> = Vec::with_capacity(bounding_boxes.len());
	for bounding_box in bounding_boxes
	{
		bounding_boxes_sorted.push(*bounding_box);
	}

	bounding_boxes_sorted.sort_by(|a, b|
	{
		let distance_a = ((a.min.add(&a.max)).divf(2.0)).sub(position);
		let distance_b = ((b.min.add(&b.max)).divf(2.0)).sub(position);
		distance_a.magnitude().partial_cmp(&distance_b.magnitude()).unwrap()
	});

	bounding_boxes_sorted[0]
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct RayHitResult
{
	pub intersects: bool,
	pub t_min: f32,
	pub t_max: f32
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Ray
{
	pub origin: Vector3,
	pub direction: Vector3
}

impl Ray
{
	pub fn new(origin: Vector3, direction: Vector3) -> Self
	{
		Self { origin, direction }
	}

pub fn intersects_aabb(&self, bounding_box: &AxisAlignedBoundingBox) -> RayHitResult
{
	let mut t_min: f32 = 0.0;
	let mut t_max: f32 = std::f32::INFINITY;

	for i in 0..3 as usize
	{
		let inv_d = 1.0 / self.direction[i];
		let mut t0 = (bounding_box.min[i] - self.origin[i]) * inv_d;
		let mut t1 = (bounding_box.max[i] - self.origin[i]) * inv_d;

		if inv_d < 0.0
		{
			std::mem::swap(&mut t0, &mut t1);
		}

		t_min = t_min.max(t0);
		t_max = t_max.min(t1);

		if t_max <= t_min
		{
			return RayHitResult { intersects: false, t_min: 0.0, t_max: 0.0 };
		}
	}

	RayHitResult { intersects: true, t_min, t_max }
}
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Plane
{
	pub normal: Vector3,
	pub distance: f32
}

impl Default for Plane
{
	fn default() -> Self
	{
		Self { normal: Vector3::zero(), distance: 0.0 }
	}
}

impl Plane
{
	pub fn new(normal: Vector3, distance: f32) -> Self
	{
		Self { normal, distance }
	}

	pub fn from_point_normal(point: Vector3, normal: Vector3) -> Self
	{
		let distance: f32 = -normal.x * point.x - normal.y * point.y - normal.z * point.z;
        Self { normal, distance }
	}
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Frustum
{
	planes:	[Vector4; 6],
}

impl Frustum
{
	pub fn new() -> Self
	{
		Self { planes: [Vector4::zero(); 6] }
	}

	pub fn is_point_inside(&self, point: &Vector3) -> bool
	{
		for plane in self.planes
		{
			let d: f32 = point.dot(&plane.xyz()) + plane.w;
			if d > 0.0
			{
				return false;
			}
		}

		true
	}

	pub fn is_aabb_inside(&self, bounding_box: &AxisAlignedBoundingBox) -> bool
	{
		todo!()
	}
}