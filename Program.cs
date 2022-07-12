using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace AnyFileRNN
{
    class Program
    {
        #region Metadata

        const int numberOfWords = 20 - 3;
        const int hiddenDim = 300;
        const double learningRate = 0.0005;
        const int epochCount = 100;
        const int seqLen = 20;
        //const string path = "..\\..\\dataset\\tawiki-20180801-pages-articles-multistream.xml";
        //const string path = "..\\..\\dataset\\enwik8";
        const string path = @"C:\Users\di_eg\OneDrive\Documents\RepoCSharp\RNNSIMPLE\rnn-master\rnn-master\dataset\text.txt";
        //const string path = "..\\..\\dataset\\abc.txt";
        //const string path = @"C:\Users\di_eg\OneDrive\Documents\RepoCSharp\RNNSIMPLE\rnn-master\rnn-master\dataset\equations.txt";
        const string state = "state.xml";
        const string output = "output.txt";

        #endregion

        static void Main(string[] args)
        {
            var dictionary = new Dictionary<string, int>();
            var writer = new StreamWriter(output);

            // CREATING VOCABULARY
            using (var reader = new StreamReader(path))
            {
                while (!reader.EndOfStream)
                {
                    var tokenizer = new Tokenizer(reader.ReadLine());
                    foreach (var token in tokenizer)
                    {
                        if (string.IsNullOrEmpty(token))
                            continue;
                        if (dictionary.ContainsKey(token))
                            dictionary[token]++;
                        else
                            dictionary.Add(token, 1);
                    }
                }
            }

            var vocab = new Vocabulary(dictionary.OrderByDescending(s => s.Value).Take(numberOfWords).Select(s => s.Key));
            RNN rnn = new RNN(vocab, vocab.Size, vocab.Size, hiddenDim, learningRate);

            // PREPARING TRAINING DATA
            var data = new DataHolder(path, vocab, seqLen);

            // TRAINING
            //rnn.Load(state);
           // Console.ReadLine();
            //rnn.ForwProp();

            for (int i = 0; i < 100; i++)
            {
                rnn.TrainEpochWithLossWriting(epochCount, data);
                writer.WriteLine($"{(i + 1) * (epochCount + 1)}____________________________________________________");
                rnn.TrainEpochWithOutput(writer, data);
            }

            rnn.Save(state);
            writer.Close();
            Console.ReadLine();
        }
    }
}