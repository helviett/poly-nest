using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using ClipperLib;

namespace PolyNester
{
	using Ngon = List<IntPoint>;
	using Ngons = List<List<IntPoint>>;

	public enum NFPQUALITY { Simple, Convex, ConcaveLight, ConcaveMedium, ConcaveHigh, ConcaveFull, Full }

	public struct Vector64
	{
		public double X;
		public double Y;
		public Vector64(double X, double Y)
		{
			this.X = X; this.Y = Y;
		}

		public Vector64(Vector64 pt)
		{
			this.X = pt.X; this.Y = pt.Y;
		}

		public static Vector64 operator +(Vector64 a, Vector64 b)
		{
			return new Vector64(a.X + b.X, a.Y + b.Y);
		}

		public static Vector64 operator -(Vector64 a, Vector64 b)
		{
			return new Vector64(a.X - b.X, a.Y - b.Y);
		}

		public static Vector64 operator *(Vector64 a, Vector64 b)
		{
			return new Vector64(a.X * b.X, a.Y * b.Y);
		}

		public static Vector64 operator /(Vector64 a, Vector64 b)
		{
			return new Vector64(a.X / b.X, a.Y / b.Y);
		}

		public static Vector64 operator *(Vector64 a, double b)
		{
			return new Vector64(a.X * b, a.Y * b);
		}

		public static Vector64 operator *(double b, Vector64 a)
		{
			return new Vector64(a.X * b, a.Y * b);
		}

		public static bool operator ==(Vector64 a, Vector64 b)
		{
			return a.X == b.X && a.Y == b.Y;
		}

		public static bool operator !=(Vector64 a, Vector64 b)
		{
			return a.X != b.X || a.Y != b.Y;
		}

		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (obj is Vector64)
			{
				Vector64 a = (Vector64)obj;
				return (X == a.X) && (Y == a.Y);
			}
			else return false;
		}

		public override int GetHashCode()
		{
			return (X.GetHashCode() ^ Y.GetHashCode());
		}
	}

	public struct Rect64
	{
		public double left;
		public double right;
		public double top;
		public double bottom;

		public Rect64(double l, double t, double r, double b)
		{
			left = l;
			right = r;
			top = t;
			bottom = b;
		}

		public double Width()
		{
			return Math.Abs(left - right);
		}

		public double Height()
		{
			return Math.Abs(top - bottom);
		}

		public double Area()
		{
			return Width() * Height();
		}

		public double Aspect()
		{
			return Width() / Height();
		}
	}

	public struct Mat3x3
	{
		public double X11, X12, X13, X21, X22, X23, X31, X32, X33;

		public static Mat3x3 Eye()
		{
			return Scale(1, 1, 1);
		}

		public static Mat3x3 RotateCounterClockwise(double t)
		{
			Mat3x3 T = new Mat3x3();

			double c = Math.Cos(t);
			double s = Math.Sin(t);

			T.X11 = c;
			T.X12 = -s;
			T.X13 = 0;
			T.X21 = s;
			T.X22 = c;
			T.X23 = 0;
			T.X31 = 0;
			T.X32 = 0;
			T.X33 = 1;

			return T;
		}

		public static Mat3x3 Scale(double scale_x, double scale_y, double scale_z = 1.0)
		{
			Mat3x3 I = new Mat3x3();
			I.X11 = scale_x;
			I.X12 = 0;
			I.X13 = 0;
			I.X21 = 0;
			I.X22 = scale_y;
			I.X23 = 0;
			I.X31 = 0;
			I.X32 = 0;
			I.X33 = scale_z;

			return I;
		}

		public static Mat3x3 Translate(double t_x, double t_y)
		{
			Mat3x3 I = new Mat3x3();
			I.X11 = 1;
			I.X12 = 0;
			I.X13 = t_x;

			I.X21 = 0;
			I.X22 = 1;
			I.X23 = t_y;
			I.X31 = 0;
			I.X32 = 0;
			I.X33 = 1;

			return I;
		}

		public static Mat3x3 operator *(Mat3x3 A, Mat3x3 B)
		{
			Mat3x3 I = new Mat3x3();
			I.X11 = A.X11 * B.X11 + A.X12 * B.X21 + A.X13 * B.X31;
			I.X12 = A.X11 * B.X12 + A.X12 * B.X22 + A.X13 * B.X32;
			I.X13 = A.X11 * B.X13 + A.X12 * B.X23 + A.X13 * B.X33;
			I.X21 = A.X21 * B.X11 + A.X22 * B.X21 + A.X23 * B.X31;
			I.X22 = A.X21 * B.X12 + A.X22 * B.X22 + A.X23 * B.X32;
			I.X23 = A.X21 * B.X13 + A.X22 * B.X23 + A.X23 * B.X33;
			I.X31 = A.X31 * B.X11 + A.X32 * B.X21 + A.X33 * B.X31;
			I.X32 = A.X31 * B.X12 + A.X32 * B.X22 + A.X33 * B.X32;
			I.X33 = A.X31 * B.X13 + A.X32 * B.X23 + A.X33 * B.X33;

			return I;
		}

		public static Vector64 operator *(Mat3x3 A, Vector64 B)
		{
			Vector64 v = new Vector64();
			v.X = A.X11 * B.X + A.X12 * B.Y + A.X13;
			v.Y = A.X21 * B.X + A.X22 * B.Y + A.X23;

			return v;
		}

		public static IntPoint operator *(Mat3x3 A, IntPoint B)
		{
			double x = A.X11 * B.X + A.X12 * B.Y + A.X13;
			double y = A.X21 * B.X + A.X22 * B.Y + A.X23;

			return new IntPoint(x, y);
		}

		private double Det2x2(double a11, double a12, double a21, double a22)
		{
			return a11 * a22 - a12 * a21;
		}

		public double Determinant()
		{
			return X11 * Det2x2(X22, X23, X32, X33) - X12 * Det2x2(X21, X23, X31, X33) + X13 * Det2x2(X21, X22, X31, X32);
		}

		public Mat3x3 Inverse()
		{
			double D = Determinant();

			Mat3x3 I = new Mat3x3();
			I.X11 = Det2x2(X22, X23, X32, X33) / D;
			I.X12 = Det2x2(X13, X12, X33, X32) / D;
			I.X13 = Det2x2(X12, X13, X22, X23) / D;
			I.X21 = Det2x2(X23, X21, X33, X31) / D;
			I.X22 = Det2x2(X11, X13, X31, X33) / D;
			I.X23 = Det2x2(X13, X11, X23, X21) / D;
			I.X31 = Det2x2(X21, X22, X31, X32) / D;
			I.X32 = Det2x2(X12, X11, X32, X31) / D;
			I.X33 = Det2x2(X11, X12, X21, X22) / D;

			return I;
		}
	}

	public static class ConverterUtility
	{
		public static IntPoint ToIntPoint(this Vector64 vec) { return new IntPoint(vec.X, vec.Y); }
		public static Vector64 ToVector64(this IntPoint vec) { return new Vector64(vec.X, vec.Y); }
		public static IntRect ToIntRect(this Rect64 rec) { return new IntRect((long)rec.left, (long)rec.top, (long)rec.right, (long)rec.bottom); }
		public static Rect64 ToRect64(this IntRect rec) { return new Rect64(rec.left, rec.top, rec.right, rec.bottom); }
	}

	public static class GeomUtility
	{
		private class PolarComparer : IComparer
		{
			public static int CompareIntPoint(IntPoint A, IntPoint B)
			{
				long det = A.Y * B.X - A.X * B.Y;

				if (det == 0)
				{
					long dot = A.X * B.X + A.Y * B.Y;
					if (dot >= 0)
						return 0;
				}

				if (A.Y == 0 && A.X > 0)
					return -1;
				if (B.Y == 0 && B.X > 0)
					return 1;
				if (A.Y > 0 && B.Y < 0)
					return -1;
				if (A.Y < 0 && B.Y > 0)
					return 1;
				return det > 0 ? 1 : -1;
			}

			int IComparer.Compare(object a, object b)
			{
				IntPoint A = (IntPoint)a;
				IntPoint B = (IntPoint)b;

				return CompareIntPoint(A, B);
			}
		}

		public static long Width(this IntRect rect)
		{
			return Math.Abs(rect.left - rect.right);
		}

		public static long Height(this IntRect rect)
		{
			return Math.Abs(rect.top - rect.bottom);
		}

		public static long Area(this IntRect rect)
		{
			return rect.Width() * rect.Height();
		}

		public static double Aspect(this IntRect rect)
		{
			return ((double)rect.Width()) / rect.Height();
		}

		public static Ngon Clone(this Ngon poly)
		{
			return new Ngon(poly);
		}

		public static Ngon Clone(this Ngon poly, long shift_x, long shift_y, bool flip_first = false)
		{
			long scale = flip_first ? -1 : 1;

			Ngon clone = new Ngon(poly.Count);
			for (int i = 0; i < poly.Count; i++)
				clone.Add(new IntPoint(scale * poly[i].X + shift_x, scale * poly[i].Y + shift_y));
			return clone;
		}

		public static Ngon Clone(this Ngon poly, Mat3x3 T)
		{
			Ngon clone = new Ngon(poly.Count);
			for (int i = 0; i < poly.Count; i++)
				clone.Add(T * poly[i]);
			return clone;
		}

		public static Ngons Clone(this Ngons polys)
		{
			Ngons clone = new Ngons(polys.Count);
			for (int i = 0; i < polys.Count; i++)
				clone.Add(polys[i].Clone());
			return clone;
		}

		public static Ngons Clone(this Ngons polys, long shift_x, long shift_y, bool flip_first = false)
		{
			Ngons clone = new Ngons(polys.Count);
			for (int i = 0; i < polys.Count; i++)
				clone.Add(polys[i].Clone(shift_x, shift_y, flip_first));
			return clone;
		}

		public static Ngons Clone(this Ngons polys, Mat3x3 T)
		{
			Ngons clone = new Ngons(polys.Count);
			for (int i = 0; i < polys.Count; i++)
				clone.Add(polys[i].Clone(T));
			return clone;
		}

		public static IntRect GetBounds(IEnumerable<IntPoint> points)
		{
			long width_min = long.MaxValue;
			long width_max = long.MinValue;
			long height_min = long.MaxValue;
			long height_max = long.MinValue;

			foreach (IntPoint p in points)
			{
				width_min = Math.Min(width_min, p.X);
				height_min = Math.Min(height_min, p.Y);
				width_max = Math.Max(width_max, p.X);
				height_max = Math.Max(height_max, p.Y);
			}

			return new IntRect(width_min, height_max, width_max, height_min);
		}

		public static Rect64 GetBounds(IEnumerable<Vector64> points)
		{
			double width_min = double.MaxValue;
			double width_max = double.MinValue;
			double height_min = double.MaxValue;
			double height_max = double.MinValue;

			foreach (Vector64 p in points)
			{
				width_min = Math.Min(width_min, p.X);
				height_min = Math.Min(height_min, p.Y);
				width_max = Math.Max(width_max, p.X);
				height_max = Math.Max(height_max, p.Y);
			}

			return new Rect64(width_min, height_max, width_max, height_min);
		}

		public static void GetRefitTransform(IEnumerable<Vector64> points, Rect64 target, bool stretch, out Vector64 scale, out Vector64 shift)
		{
			Rect64 bds = GetBounds(points);

			scale = new Vector64(target.Width() / bds.Width(), target.Height() / bds.Height());

			if (!stretch)
			{
				double s = Math.Min(scale.X, scale.Y);
				scale = new Vector64(s, s);
			}

			shift = new Vector64(-bds.left, -bds.bottom) * scale
				+ new Vector64(Math.Min(target.left, target.right), Math.Min(target.bottom, target.top));
		}

		public static Ngon ConvexHull(Ngon subject, double rigidness = 0)
		{
			if (subject.Count == 0)
				return new Ngon();

			if (rigidness >= 1)
				return subject.Clone();

			subject = subject.Clone();
			if (Clipper.Area(subject) < 0)
				Clipper.ReversePaths(new Ngons() { subject });

			Ngon last_hull = new Ngon();
			Ngon hull = subject;

			double subj_area = Clipper.Area(hull);

			int last_vert = 0;
			for (int i = 1; i < subject.Count; i++)
				if (hull[last_vert].Y > hull[i].Y)
					last_vert = i;

			while (last_hull.Count != hull.Count)
			{
				last_hull = hull;
				hull = new Ngon();
				hull.Add(last_hull[last_vert]);

				int steps_since_insert = 0;
				int max_steps = rigidness <= 0 ? int.MaxValue : (int)Math.Round(10 - (10 * rigidness));

				int n = last_hull.Count;

				int start = last_vert;
				for (int i = 1; i < n; i++)
				{
					IntPoint a = last_hull[last_vert];
					IntPoint b = last_hull[(start + i) % n];
					IntPoint c = last_hull[(start + i + 1) % n];

					IntPoint ab = new IntPoint(b.X - a.X, b.Y - a.Y);
					IntPoint ac = new IntPoint(c.X - a.X, c.Y - a.Y);

					if (ab.Y * ac.X < ab.X * ac.Y || steps_since_insert >= max_steps)
					{
						hull.Add(b);
						last_vert = (start + i) % n;
						steps_since_insert = -1;
					}
					steps_since_insert++;
				}

				last_vert = 0;

				double hull_area = Clipper.Area(hull);

				if (subj_area / hull_area < Math.Sqrt(rigidness))
				{
					hull = Clipper.SimplifyPolygon(hull, PolyFillType.pftNonZero)[0];
					break;
				}
			}

			return hull;
		}

		public static Ngons MinkowskiSumSegment(Ngon pattern, IntPoint p1, IntPoint p2, bool flip_pattern)
		{
			Clipper clipper = new Clipper();

			Ngon p1_c = pattern.Clone(p1.X, p1.Y, flip_pattern);

			if (p1 == p2)
				return new Ngons() { p1_c };

			Ngon p2_c = pattern.Clone(p2.X, p2.Y, flip_pattern);

			Ngons full = new Ngons();
			clipper.AddPath(p1_c, PolyType.ptSubject, true);
			clipper.AddPath(p2_c, PolyType.ptSubject, true);
			clipper.AddPaths(Clipper.MinkowskiSum(pattern.Clone(0, 0, flip_pattern), new Ngon() { p1, p2 }, false), PolyType.ptSubject, true);
			clipper.Execute(ClipType.ctUnion, full, PolyFillType.pftNonZero);

			return full;
		}

		public static Ngons MinkowskiSumBoundary(Ngon pattern, Ngon path, bool flip_pattern)
		{
			Clipper clipper = new Clipper();

			Ngons full = new Ngons();

			for (int i = 0; i < path.Count; i++)
			{
				IntPoint p1 = path[i];
				IntPoint p2 = path[(i + 1) % path.Count];

				Ngons seg = MinkowskiSumSegment(pattern, p1, p2, flip_pattern);
				clipper.AddPaths(full, PolyType.ptSubject, true);
				clipper.AddPaths(seg, PolyType.ptSubject, true);

				Ngons res = new Ngons();
				clipper.Execute(ClipType.ctUnion, res, PolyFillType.pftNonZero);
				full = res;
				clipper.Clear();
			}

			return full;
		}

		public static Ngons MinkowskiSumBoundary(Ngon pattern, Ngons path, bool flip_pattern)
		{
			Clipper clipper = new Clipper();

			Ngons full = new Ngons();

			for (int i = 0; i < path.Count; i++)
			{
				Ngons seg = MinkowskiSumBoundary(pattern, path[i], flip_pattern);
				clipper.AddPaths(full, PolyType.ptSubject, true);
				clipper.AddPaths(seg, PolyType.ptSubject, true);

				Ngons res = new Ngons();
				clipper.Execute(ClipType.ctUnion, res, PolyFillType.pftNonZero);
				full = res;
				clipper.Clear();
			}

			return full;
		}

		private static Ngons MSumSimple(Ngon pattern, Ngons subject, bool flip_pattern)
		{
			IntRect pB = GetBounds(pattern);
			IntRect sB = GetBounds(subject[0]);

			if (flip_pattern)
			{
				pB = new IntRect(-pB.right, -pB.bottom, -pB.left, -pB.top);
			}

			long l = pB.left + sB.left;
			long r = pB.right + sB.right;
			long t = pB.top + sB.top;
			long b = pB.bottom + sB.bottom;

			Ngon p = new Ngon() { new IntPoint(l, b), new IntPoint(r, b), new IntPoint(r, t), new IntPoint(l, t) };
			return new Ngons() { p };
		}

		private static Ngons MSumConvex(Ngon pattern, Ngons subject, bool flip_pattern)
		{
			Ngon h_p = ConvexHull(pattern.Clone(0, 0, flip_pattern));
			Ngon h_s = ConvexHull(subject[0].Clone());

			int n_p = h_p.Count;
			int n_s = h_s.Count;

			int sp = 0;
			for (int k = 0; k < n_p; k++)
				if (h_p[k].Y < h_p[sp].Y)
					sp = k;

			int ss = 0;
			for (int k = 0; k < n_s; k++)
				if (h_s[k].Y < h_s[ss].Y)
					ss = k;

			Ngon poly = new Ngon(n_p + n_s);

			int i = 0;
			int j = 0;
			while (i < n_p || j < n_s)
			{
				int ip = (sp + i + 1) % n_p;
				int jp = (ss + j + 1) % n_s;
				int ii = (sp + i) % n_p;
				int jj = (ss + j) % n_s;

				IntPoint sum = new IntPoint(h_p[ii].X + h_s[jj].X, h_p[ii].Y + h_s[jj].Y);
				IntPoint v = new IntPoint(h_p[ip].X - h_p[ii].X, h_p[ip].Y - h_p[ii].Y);
				IntPoint w = new IntPoint(h_s[jp].X - h_s[jj].X, h_s[jp].Y - h_s[jj].Y);

				poly.Add(sum);

				if (i == n_p)
				{
					j++;
					continue;
				}

				if (j == n_s)
				{
					i++;
					continue;
				}

				long cross = v.Y * w.X - v.X * w.Y;

				if (cross < 0) i++;
				else if (cross > 0) j++;
				else
				{
					long dot = v.X * w.X + v.Y * w.Y;
					if (dot > 0)
					{
						i++;
						j++;
					}
					else
					{
						throw new Exception();
					}
				}
			}

			return Clipper.SimplifyPolygon(poly);
		}

		private static Ngons MSumConcave(Ngon pattern, Ngons subject, bool flip_pattern, double rigidness = 1.0)
		{
			Ngon subj = subject[0];
			Ngon patt = pattern.Clone(0, 0, flip_pattern);

			if (rigidness < 1.0)
			{
				subj = ConvexHull(subj, rigidness);
				patt = ConvexHull(patt, rigidness);
			}

			Ngons sres = MinkowskiSumBoundary(patt, subj, false);
			return sres.Count == 0 ? sres : new Ngons() { sres[0] };
		}

		private static Ngons MSumFull(Ngon pattern, Ngons subject, bool flip_pattern)
		{
			Clipper clipper = new Clipper();

			Ngons full = new Ngons();

			long scale = flip_pattern ? -1 : 1;

			for (int i = 0; i < pattern.Count; i++)
				clipper.AddPaths(subject.Clone(scale * pattern[i].X, scale * pattern[i].Y), PolyType.ptSubject, true);

			clipper.Execute(ClipType.ctUnion, full, PolyFillType.pftNonZero);
			clipper.Clear();

			clipper.AddPaths(full, PolyType.ptSubject, true);
			clipper.AddPaths(MinkowskiSumBoundary(pattern, subject, flip_pattern), PolyType.ptSubject, true);

			Ngons res = new Ngons();

			clipper.Execute(ClipType.ctUnion, res, PolyFillType.pftNonZero);

			return res;
		}

		public static Ngons MinkowskiSum(Ngon pattern, Ngons subject, NFPQUALITY quality, bool flip_pattern)
		{
			switch (quality)
			{
				case NFPQUALITY.Simple:
					return MSumSimple(pattern, subject, flip_pattern);
				case NFPQUALITY.Convex:
					return MSumConvex(pattern, subject, flip_pattern);
				case NFPQUALITY.ConcaveLight:
					return MSumConcave(pattern, subject, flip_pattern, 0.25);
				case NFPQUALITY.ConcaveMedium:
					return MSumConcave(pattern, subject, flip_pattern, 0.55);
				case NFPQUALITY.ConcaveHigh:
					return MSumConcave(pattern, subject, flip_pattern, 0.85);
				case NFPQUALITY.ConcaveFull:
					return MSumConcave(pattern, subject, flip_pattern, 1.0);
				case NFPQUALITY.Full:
					return MSumFull(pattern, subject, flip_pattern);
				default:
					return null;
			}
		}

		public static Ngon CanFitInsidePolygon(IntRect canvas, Ngon pattern)
		{
			IntRect bds = GetBounds(pattern);

			long l = canvas.left - bds.left;
			long r = canvas.right - bds.right;
			long t = canvas.top - bds.top;
			long b = canvas.bottom - bds.bottom;

			if (l > r || b > t)
				return null;
			if (l == r && b == t)
				return new Ngon { new IntPoint(l, t), };
			if (l == r || b == t)
				return new Ngon { new IntPoint(l, b), new IntPoint(r, t), };
			return new Ngon() { new IntPoint(l, b), new IntPoint(r, b), new IntPoint(r, t), new IntPoint(l, t) };
		}

		public static double AlignToEdgeRotation(Ngon target, int edge_start)
		{
			edge_start %= target.Count;
			int next_pt = (edge_start + 1) % target.Count;
			IntPoint best_edge = new IntPoint(target[next_pt].X - target[edge_start].X, target[next_pt].Y - target[edge_start].Y);
			return -Math.Atan2(best_edge.Y, best_edge.X);
		}

		public static bool AlmostRectangle(Ngon target, double percent_diff = 0.05)
		{
			IntRect bounds = GetBounds(target);
			double area = Math.Abs(Clipper.Area(target));

			return 1.0 - area / bounds.Area() < percent_diff;
		}
	}

	public class Nester
	{
		private long upscale = 1;

		public long Upscale
		{
			get => upscale;
			set
			{
				if (value <= 0) {
					throw new InvalidOperationException("Upscale should be greate or equal to 1.");
				}
				upscale = value;
			}
		}

		public Vector64 Container { get; set; }

		private class PolyRef
		{
			public Ngons poly;
			public Ngons original;
			public Mat3x3 trans;
			public bool is_placed;

			public IntPoint GetTransformedPoint(int poly_id, int index) => trans * poly[poly_id][index];

			public IntPoint GetTransformedOriginalPoint(int poly_id, int index) => trans * original[poly_id][index];

			public Ngons GetTransformedPoly()
			{
				Ngons n = new Ngons(poly.Count);
				for (int i = 0; i < poly.Count; i++)
				{
					Ngon nn = new Ngon(poly[i].Count);
					for (int j = 0; j < poly[i].Count; j++)
						nn.Add(GetTransformedPoint(i, j));
					n.Add(nn);
				}
				return n;
			}

			public Ngons GetTransformedOriginalPoly()
			{
				Ngons n = new Ngons(original.Count);
				for (int i = 0; i < original.Count; i++) {
					Ngon nn = new Ngon(original[i].Count);
					for (int j = 0; j < original[i].Count; j++)
						nn.Add(GetTransformedOriginalPoint(i, j));
					n.Add(nn);
				}
				return n;
			}
		}

		private class Command
		{
			public Action<object[]> Call;
			public object[] param;
		}

		private const long unit_scale = 10000000;

		private List<PolyRef> polygon_lib;  // list of saved polygons for reference by handle, stores raw poly positions and transforms

		public int LibSize { get { return polygon_lib.Count; } }

		public void RemovePlaced() => polygon_lib.RemoveAll(pr => pr.is_placed);

		public Nester()
		{
			polygon_lib = new List<PolyRef>();
		}

		private HashSet<int> PreprocessHandles(IEnumerable<int> handles)
		{
			if (handles == null)
				handles = Enumerable.Range(0, polygon_lib.Count);

			HashSet<int> unique = new HashSet<int>();
			foreach (int i in handles)
				unique.Add(i);

			return unique;
		}

		public void Scale(int handle, double scale_x, double scale_y)
		{
			polygon_lib[handle].trans = Mat3x3.Scale(scale_x, scale_y) * polygon_lib[handle].trans;
		}

		public void Rotate(int handle, double theta)
		{
			polygon_lib[handle].trans = Mat3x3.RotateCounterClockwise(theta) * polygon_lib[handle].trans;
		}

		public void Translate(int handle, double translate_x, double translate_y)
		{
			polygon_lib[handle].trans = Mat3x3.Translate(translate_x, translate_y) * polygon_lib[handle].trans;
		}

		public void TranslateOriginToZero(IEnumerable<int> handles)
		{
			HashSet<int> unique = PreprocessHandles(handles);

			foreach (int i in unique) {
				IntPoint o = polygon_lib[i].GetTransformedPoint(0, 0);
				Translate(i, -o.X, -o.Y);
			}
		}

		public void Refit(Rect64 target, bool stretch, IEnumerable<int> handles)
		{
			HashSet<int> unique = PreprocessHandles(handles);

			HashSet<Vector64> points = new HashSet<Vector64>();
			foreach (int i in unique)
				points.UnionWith(polygon_lib[i].poly[0].Select(p => polygon_lib[i].trans * new Vector64(p.X, p.Y)));

			Vector64 scale, trans;
			GeomUtility.GetRefitTransform(points, target, stretch, out scale, out trans);

			foreach (int i in unique) {
				Scale(i, scale.X, scale.Y);
				Translate(i, trans.X, trans.Y);
			}
		}

		/// <summary>
		/// Get the optimal quality for tradeoff between speed and precision of NFP
		/// </summary>
		/// <param name="subj_handle"></param>
		/// <param name="pattern_handle"></param>
		/// <returns></returns>
		private NFPQUALITY GetNFPQuality(int subj_handle, int pattern_handle, double max_area_bounds)
		{
			Ngon S = polygon_lib[subj_handle].GetTransformedPoly()[0];
			Ngon P = polygon_lib[pattern_handle].GetTransformedPoly()[0];

			if (GeomUtility.AlmostRectangle(S) && GeomUtility.AlmostRectangle(P))
				return NFPQUALITY.Simple;

			double s_A = GeomUtility.GetBounds(S).Area();
			double p_A = GeomUtility.GetBounds(P).Area();

			if (p_A / s_A > 1000)
				return NFPQUALITY.Simple;

			if (s_A / max_area_bounds < 0.05)
				return NFPQUALITY.Simple;

			if (p_A / s_A > 100)
				return NFPQUALITY.Convex;

			if (p_A / s_A > 50)
				return NFPQUALITY.ConcaveLight;

			if (p_A / s_A > 10)
				return NFPQUALITY.ConcaveMedium;

			if (p_A / s_A > 2)
				return NFPQUALITY.ConcaveHigh;

			if (p_A / s_A > 0.25)
				return NFPQUALITY.ConcaveFull;

			return NFPQUALITY.Full;
		}

		/// <summary>
		/// Parallel kernel for generating NFP of pattern on handle, return the index in the library of this NFP
		/// Decides the optimal quality for this NFP
		/// </summary>
		/// <param name="subj_handle"></param>
		/// <param name="pattern_handle"></param>
		/// <param name="lib_set_at"></param>
		/// <returns></returns>
		private int NFPKernel(int subj_handle, int pattern_handle, double max_area_bounds, int lib_set_at, NFPQUALITY max_quality = NFPQUALITY.Full)
		{
			NFPQUALITY quality = GetNFPQuality(subj_handle, pattern_handle, max_area_bounds);
			quality = (NFPQUALITY)Math.Min((int)quality, (int)max_quality);
			return AddMinkowskiSum(subj_handle, pattern_handle, quality, true, lib_set_at);
		}

		/// <summary>
		/// Regular for loop in the syntax of a parallel for used for debugging
		/// </summary>
		/// <param name="i"></param>
		/// <param name="j"></param>
		/// <param name="body"></param>
		private void For(int i, int j, Action<int> body)
		{
			for (int k = i; k < j; k++)
				body(k);
		}

		/// <summary>
		/// Nest the collection of handles with minimal enclosing square from origin
		/// </summary>
		/// <param name="handles"></param>
		public void Nest(IEnumerable<int> handles, NFPQUALITY max_quality = NFPQUALITY.Full)
		{
			HashSet<int> unique = PreprocessHandles(handles);

			TranslateOriginToZero(unique);

			int n = unique.Count;

			Dictionary<int, IntRect> bounds = new Dictionary<int, IntRect>();
			foreach (int handle in unique)
				bounds.Add(handle, GeomUtility.GetBounds(polygon_lib[handle].GetTransformedPoly()[0]));

			int[] ordered_handles = unique.OrderByDescending(p => Math.Max(bounds[p].Height(), bounds[p].Width())).ToArray();
			double max_bound_area = bounds[ordered_handles[0]].Area();

			int start_cnt = polygon_lib.Count;

			int[] canvas_regions = AddCanvasFitPolygon(ordered_handles);

			int base_cnt = polygon_lib.Count;
			for (int i = 0; i < n * n - n; i++)
				polygon_lib.Add(new PolyRef());

			int update_breaks = 10;
			int nfp_chunk_sz = n * n / update_breaks * update_breaks == n * n ? n * n / update_breaks : n * n / update_breaks + 1;

			// the row corresponds to pattern and col to nfp for this pattern on col subj
			int[,] nfps = new int[n, n];
			for (int k = 0; k < update_breaks; k++) {
				int start = k * nfp_chunk_sz;
				int end = Math.Min((k + 1) * nfp_chunk_sz, n * n);

				if (start >= end)
					break;

				Parallel.For(start, end, i => nfps[i / n, i % n] = i / n == i % n ? -1 : NFPKernel(ordered_handles[i % n], ordered_handles[i / n], max_bound_area, base_cnt + i - (i % n > i / n ? 1 : 0) - i / n, max_quality));

				double progress = Math.Min(((double)(k + 1)) / (update_breaks + 1) * 50.0, 50.0);
			}

			int place_chunk_sz = Math.Max(n / update_breaks, 1);

			bool[] placed = new bool[n];
			for (int i = 0; i < n; i++) {
				Clipper c = new Clipper();
				var canvas = polygon_lib[canvas_regions[i]].poly[0];
				IntPoint place = new IntPoint(0, 0);
				IntPoint o = polygon_lib[ordered_handles[i]].GetTransformedPoint(0, 0);
				if (canvas.Count == 0)
					continue;
				else if (canvas.Count == 1) {
					// One point region: check if any nfp contains that point
					// According to current implementation this branch can be simplified to:
					// `placed[i] = !placed.Any(p => p);`
					// But we'll generalize it.
					place = canvas[0];
					placed[i] = true;
					for (int j = 0; j < i; j++) {
						if (!placed[j])
							continue;
						var nfp = polygon_lib[nfps[i, j]].GetTransformedPoly();
						if (IsPointInPolygon(place, nfp)) {
							placed[i] = false;
							break;
						}
					}
				} else if (canvas.Count == 2) {
					c.AddPath(canvas, PolyType.ptSubject, false);
					for (int j = 0; j < i; j++) {
						if (!placed[j])
							continue;

						c.AddPaths(polygon_lib[nfps[i, j]].GetTransformedPoly(), PolyType.ptClip, true);
					}
					var fit_region = new PolyTree();
					c.Execute(ClipType.ctDifference, fit_region, PolyFillType.pftNonZero);
					IntRect bds = bounds[ordered_handles[i]];
					long ext_x = bds.right - o.X;
					long ext_y = bds.top - o.Y;
					long pl_score = long.MaxValue;
					var node = (PolyNode)fit_region;
					while (node != null) {
						foreach (var vertex in node.Contour) {
							long cd_score = Math.Max(vertex.X + ext_x, vertex.Y + ext_y);
							if (cd_score < pl_score) {
								pl_score = cd_score;
								place = vertex;
								placed[i] = true;
							}
						}
						node = node.GetNext();
					}
				} else {
					// Copy-paster because filling and iterating ngons is 10-15% faster then PolyTree
					c.AddPath(canvas, PolyType.ptSubject, true);
					for (int j = 0; j < i; j++) {
						if (!placed[j])
							continue;

						c.AddPaths(polygon_lib[nfps[i, j]].GetTransformedPoly(), PolyType.ptClip, true);
					}
					Ngons fit_region = new Ngons();
					c.Execute(ClipType.ctDifference, fit_region, PolyFillType.pftNonZero);
					IntRect bds = bounds[ordered_handles[i]];
					long ext_x = bds.right - o.X;
					long ext_y = bds.top - o.Y;
					long pl_score = long.MaxValue;
					for (int k = 0; k < fit_region.Count; k++)
						for (int l = 0; l < fit_region[k].Count; l++) {
							IntPoint cand = fit_region[k][l];
							long cd_score = Math.Max(cand.X + ext_x, cand.Y + ext_y);
							if (cd_score < pl_score) {
								pl_score = cd_score;
								place = cand;
								placed[i] = true;
							}
						}
				}
				if (!placed[i])
					continue;

				Translate(ordered_handles[i], (double)(place.X - o.X), (double)(place.Y - o.Y));
				for (int k = i + 1; k < n; k++)
					Translate(nfps[k, i], (double)(place.X - o.X), (double)(place.Y - o.Y));
			}

			for (int i = 0; i < ordered_handles.Length; i++) {
				polygon_lib[ordered_handles[i]].is_placed = placed[i];
			}

			// remove temporary added values
			polygon_lib.RemoveRange(start_cnt, polygon_lib.Count - start_cnt);
		}

		public void FindOptimalRotation(IEnumerable<int> handles)
		{
			HashSet<int> unique = PreprocessHandles(handles);

			foreach (int i in unique)
				FindOptimalRotation(i);
		}

		private void FindOptimalRotation(int handle)
		{
			Ngon hull = polygon_lib[handle].GetTransformedPoly()[0];
			int n = hull.Count;

			double best_t = 0;
			int best = 0;
			long best_area = long.MaxValue;
			bool flip_best = false;

			for (int i = 0; i < n; i++)
			{
				double t = GeomUtility.AlignToEdgeRotation(hull, i);

				Mat3x3 rot = Mat3x3.RotateCounterClockwise(t);

				Ngon clone = hull.Clone(rot);

				IntRect bounds = GeomUtility.GetBounds(clone);
				long area = bounds.Area();
				double aspect = bounds.Aspect();

				if (area < best_area)
				{
					best_area = area;
					best = i;
					best_t = t;
					flip_best = aspect > 1.0;
				}
			}

			double flip = flip_best ? Math.PI * 0.5 : 0;
			IntPoint around = hull[best];

			Translate(handle, -around.X, -around.Y);
			Rotate(handle, best_t + flip);
			Translate(handle, around.X, around.Y);
		}

		public int AddPolygon(List<List<Vector64>> polygon, bool simplify = false)
		{
			var clipper_polygon = polygon.Select(part =>
				part.Select(v =>
					new IntPoint(v.X * upscale, v.Y * upscale)
				).ToList()
			).ToList();
			if (simplify) {
				clipper_polygon = Clipper.SimplifyPolygons(clipper_polygon, PolyFillType.pftNonZero);
			}
			polygon_lib.Add(new PolyRef {
				poly = clipper_polygon,
				original = clipper_polygon.Clone(),
				trans = Mat3x3.Eye(),
			});
			return polygon_lib.Count - 1;
		}

		public int AddMinkowskiSum(int subj_handle, int pattern_handle, NFPQUALITY quality, bool flip_pattern, int set_at = -1)
		{
			Ngons A = polygon_lib[subj_handle].GetTransformedPoly();
			Ngons B = polygon_lib[pattern_handle].GetTransformedPoly();

			Ngons C = GeomUtility.MinkowskiSum(B[0], A, quality, flip_pattern);
			PolyRef pref = new PolyRef() { poly = C, trans = Mat3x3.Eye() };

			if (set_at < 0)
				polygon_lib.Add(pref);
			else
				polygon_lib[set_at] = pref;

			return set_at < 0 ? polygon_lib.Count - 1 : set_at;
		}

		public int AddCanvasFitPolygon(IntRect canvas, int pattern_handle)
		{
			Ngon B = polygon_lib[pattern_handle].GetTransformedOriginalPoly()[0];

			Ngon C = GeomUtility.CanFitInsidePolygon(canvas, B) ?? new Ngon();
			polygon_lib.Add(new PolyRef() { poly = new Ngons() { C }, trans = Mat3x3.Eye() });
			return polygon_lib.Count - 1;
		}

		public int AddCanvasFitPolygon(Rect64 canvas, int pattern_handle)
		{
			IntRect c = new IntRect((long)canvas.left, (long)canvas.top, (long)canvas.right, (long)canvas.bottom);
			return AddCanvasFitPolygon(c, pattern_handle);
		}

		public int[] AddCanvasFitPolygon(IEnumerable<int> handles)
		{
			HashSet<int> unique = PreprocessHandles(handles);

			long w = (long)(Container.X * Upscale);
			long h = (long)(Container.Y * Upscale);

			if (w == 0 || h == 0) {
				foreach (int i in unique) {
					IntRect bds = GeomUtility.GetBounds(polygon_lib[i].GetTransformedPoly()[0]);
					w += bds.Width();
					h += bds.Height();
				}

				w += 1000;
				h += 1000;
			}

			IntRect canvas = new IntRect(0, h, w, 0);

			return handles.Select(p => AddCanvasFitPolygon(canvas, p)).ToArray();
		}

		public Ngons GetTransformedPoly(int handle)
		{
			return polygon_lib[handle].GetTransformedPoly();
		}

		public bool IsPolygonPlaced(int handle) => polygon_lib[handle].is_placed;

		public Mat3x3 GetPolygonTransform(int handle) => polygon_lib[handle].trans;

		public void ResetTransformLib()
		{
			for (int i = 0; i < polygon_lib.Count; i++)
				polygon_lib[i].trans = Mat3x3.Eye();
		}

		public void ApplyTransformLibUVSpace(Vector64[] points, int[] handles)
		{
			for (int i = 0; i < points.Length; i++)
				points[i] = polygon_lib[handles[i]].trans * (unit_scale * points[i]);
		}

		public void RevertTransformLibUVSpace(Vector64[] points, int[] handles)
		{
			for (int i = 0; i < points.Length; i++)
				points[i] = (1.0 / unit_scale) * (polygon_lib[handles[i]].trans.Inverse() * points[i]);
		}

		public void Offset(IEnumerable<int> handles, double by)
		{
			by *= upscale;
			HashSet<int> unique = PreprocessHandles(handles);
			var clipper_offset = new ClipperOffset();
			foreach (var handle in unique) {
				var polygon = polygon_lib[handle].original;
				clipper_offset.AddPaths(polygon, JoinType.jtMiter, EndType.etClosedPolygon);
				clipper_offset.Execute(ref polygon_lib[handle].poly, by);
				clipper_offset.Clear();
			}
		}

		public void Offset(int handle, double by)
		{
			by *= upscale;
			var clipper_offset = new ClipperOffset();
			var polygon = polygon_lib[handle].poly;
			clipper_offset.AddPaths(polygon, JoinType.jtMiter, EndType.etClosedPolygon);
			clipper_offset.Execute(ref polygon_lib[handle].poly, by);
			clipper_offset.Clear();
		}

		private static bool IsPointInPolygon(IntPoint point, Ngons polygon)
		{
			var windingNumber = 0;
			foreach (var part in polygon) {
				var current = part[^1];
				foreach (var next in part) {
					var o = IntPoint.Cross(current - point, next - current);
					if (current.Y <= point.Y) {
						if (next.Y > point.Y) {
							if (o == 0) {
								return true;
							}
							if (o < 0) {
								windingNumber++;
							}
						}
					} else {
						if (next.Y <= point.Y) {
							if (o == 0) {
								return true;
							}
							if (o > 0) {
								windingNumber--;
							}
						}
					}
					current = next;
				}
			}
			return windingNumber != 0;
		}
	}
}
