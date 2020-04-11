#pragma once
#include <math.h>
#include "Vector.h"
struct Camera
{
	vec3 Position;	//позиция камеры
	vec3 View;		//направление наблюдения
	vec3 UpVector;	//вектор поворота сценs
	void SetCamera(GLfloat posX, GLfloat posY, GLfloat posZ, GLfloat viewX, GLfloat viewY, GLfloat viewZ, GLfloat upX, GLfloat upY, GLfloat upZ)
	{
		//установить позицию камеры
		vec3 _Position = vec3(posX, posY, posZ);
		vec3 _View = vec3(viewX, viewY, viewZ);
		vec3 _UpVector = vec3(upX, upY, upZ);

		Position = _Position;
		View = _View;
		UpVector = _UpVector;
	}
	void RotateView(GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
	{
		vec3 _newView;
		vec3 _View;
		_View = View - Position;

		GLfloat cosA = (GLfloat)cos(angle);
		GLfloat sinA = (GLfloat)sin(angle);
		// если все матрицы перемножить - получится это


		_newView.e[0] = (cosA + (1 - cosA) * x * x) * _View.x();
		_newView.e[0] += ((1 - cosA) * x * y - z * sinA) * _View.y();
		_newView.e[0] += ((1 - cosA) * x * z + y * sinA) * _View.z();

		_newView.e[1] = ((1 - cosA) * x * y + z * sinA) * _View.x();
		_newView.e[1] += (cosA + (1 - cosA) * y * y) * _View.y();
		_newView.e[1] += ((1 - cosA) * y * z - x * sinA) * _View.z();

		_newView.e[2] = ((1 - cosA) * x * z - y * sinA) * _View.x();
		_newView.e[2] += ((1 - cosA) * y * z + x * sinA) * _View.y();
		_newView.e[2] += (cosA + (1 - cosA) * z * z) * _View.z();


		View.e[0] = Position.x() + _newView.x();
		View.e[1] = Position.y() + _newView.y();
		View.e[2] = Position.z() + _newView.z();
	}
	void MoveCamera(GLfloat speed)
	{
		vec3 n(View - Position);
		n.make_unit_vector();
		Position += n * speed;
		View += n * speed;
	}
	void MoveCameraLeftRight(GLfloat speed)
	{

		vec3 n = -Position + View; // camDir
		n.make_unit_vector();
		vec3 u = cross(UpVector, n); //X axis
		u.make_unit_vector();
		Position += u * speed;
		View += u * speed;
	}
	void MouseView(GLint width, GLint height)			//установка вида с помощью мыши
	{
		POINT mousePos;

		GLint centrX = glutGet(GLUT_WINDOW_X) + width / 2.0f;
		GLint Y = glutGet(GLUT_WINDOW_Y) + height / 2.0f;

		GLfloat angleY = 0.0f;
		GLfloat angleZ = 0.0f;
		static GLfloat currentXRotation = 0.0f;
		GetCursorPos(&mousePos);

		if (mousePos.x == centrX && mousePos.y == Y) return;

		SetCursorPos(centrX, Y);

		angleY = (GLfloat)((centrX - mousePos.x)) / 500.0f;
		angleZ = (GLfloat)((Y - mousePos.y)) / 500.0f;

		static GLfloat lastXRotation = 0.0f;
		lastXRotation = currentXRotation;
		if (currentXRotation > 1.0f)
		{
			currentXRotation = 1.0f;
			if (lastXRotation != 1.0f)
			{
				vec3 vAxis = cross(View - Position, UpVector);
				vAxis.make_unit_vector();


				RotateView(1.0f - lastXRotation, vAxis.x(), vAxis.y(), vAxis.z());
			}
		}

		//Если угол меньше -1.0f
		else if (currentXRotation < -1.0f)
		{
			currentXRotation = -1.0f;
			if (lastXRotation != -1.0f)
			{
				//вычисляем ось
				vec3 vAxis = (cross(View - Position, UpVector));
				vAxis.make_unit_vector();
				//вращаем
				RotateView(-1.0f - lastXRotation, vAxis.x(), vAxis.y(), vAxis.z());
			}
		}

		else
		{
			vec3 vAxis = (cross(View - Position, UpVector));
			vAxis.make_unit_vector();
			RotateView(angleZ, vAxis.x(), vAxis.y(), vAxis.z());
		}
		RotateView(angleY, 0, 1, 0);
	}

};