using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.IO;
using System.Linq;

class FeatureBasedAnalysis
{
    static void Main(string[] args)
    {
        try
        {
            // Pobranie ścieżek z argumentów wiersza poleceń lub ustawienia domyślnych wartości
            string inputImagePath = args.Length > 0 ? args[0] : @"C:\Users\ardki\source\repos\Obraz_2.2\Obraz_start\in.png.png";
            string outputDirectory = args.Length > 1 ? args[1] : @"C:\Users\ardki\source\repos\Obraz_2.2\Wynik\";

            // Sprawdzanie istnienia obrazu wejściowego
            if (!File.Exists(inputImagePath))
            {
                Console.WriteLine("Podany plik wejściowy nie istnieje: " + inputImagePath);
                return;
            }

            // Tworzenie katalogu wyjściowego, jeżeli nie istnieje
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            // Wczytanie obrazu
            Mat image = CvInvoke.Imread(inputImagePath, ImreadModes.Color);

            if (image.IsEmpty)
            {
                Console.WriteLine("Obraz wejściowy jest pusty lub niepoprawny.");
                return;
            }

            Mat grayImage = new Mat();
            CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
            CvInvoke.GaussianBlur(grayImage, grayImage, new System.Drawing.Size(5, 5), 2);

            // Wykrywanie krawędzi
            Mat edges = new Mat();
            CvInvoke.Canny(grayImage, edges, 50, 150);

            // Znajdowanie konturów
            using (var contours = new Emgu.CV.Util.VectorOfVectorOfPoint())
            {
                Mat hierarchy = new Mat();
                CvInvoke.FindContours(edges, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                // Kopia obrazu.
                Mat contourImage = image.Clone();

                // Znalezienie środka obrazu
                int centerX = image.Width / 2;
                int centerY = image.Height / 2;
                var imageCenter = new System.Drawing.Point(centerX, centerY);

                // Znalezienie konturu najbliższego do środka
                double minDistance = double.MaxValue;
                int closestContourIndex = -1;

                for (int i = 0; i < contours.Size; i++)
                {
                    var contour = contours[i];
                    var moments = CvInvoke.Moments(contour);
                    if (moments.M00 != 0)
                    {
                        int cx = (int)(moments.M10 / moments.M00);
                        int cy = (int)(moments.M01 / moments.M00);
                        double distance = Math.Sqrt(Math.Pow(cx - centerX, 2) + Math.Pow(cy - centerY, 2));

                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            closestContourIndex = i;
                        }
                    }
                }

                // Zaznaczenie konturu na czerwono
                if (closestContourIndex != -1)
                {
                    CvInvoke.DrawContours(contourImage, contours, closestContourIndex, new Emgu.CV.Structure.MCvScalar(0, 0, 255), 2);
                }
                else
                {
                    Console.WriteLine("Nie znaleziono konturów na obrazie.");
                    return;
                }

                // Zapisanie obrazu z konturem
                string outputImagePath = Path.Combine(outputDirectory, "contour_closest_to_center.png");
                CvInvoke.Imwrite(outputImagePath, contourImage);

                Console.WriteLine("Obraz z zaznaczonym najbliższym konturem zapisany: " + outputImagePath);
            }


        }
        catch (Exception ex)
        {
            // Informacja o błedzie
            Console.WriteLine("Wystąpił błąd: " + ex.Message);
        }
    }
}
