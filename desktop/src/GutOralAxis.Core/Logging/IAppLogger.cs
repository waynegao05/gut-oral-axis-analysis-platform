namespace GutOralAxis.Core.Logging;

public interface IAppLogger
{
    void Information(string eventName, string message);

    void Warning(string eventName, string message);

    void Error(string eventName, string message, Exception? exception = null);
}

public sealed class NullAppLogger : IAppLogger
{
    public void Information(string eventName, string message)
    {
    }

    public void Warning(string eventName, string message)
    {
    }

    public void Error(string eventName, string message, Exception? exception = null)
    {
    }
}
